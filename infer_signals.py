# -*- coding: utf-8 -*-
"""
infer_signals.py
---------------------------------------------------------------
Runs inference over data/processed/*.feather using:
- Saved feature pipeline at models/preproc_pipeline.json
- A trained model from models/ (supports sklearn .pkl or PyTorch .pt)
Outputs per-symbol CSVs under data/signals/{SYMBOL}.csv:
  timestamp,symbol,close,score

Usage:
  python infer_signals.py
"""
import os, glob, json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from runtime_flags import get_float

from features_pipeline import FeaturePipeline

PROCESSED = Path("data/processed")
PREPROC = Path("models/preproc_pipeline.json")
MODELS_DIR = Path("models")
OUT_DIR = Path("data/signals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _load_model():
    """Load a trained model from models/ directory.

    Security Policy (fail-closed):
    - PyTorch models are loaded with weights_only=True by default
    - Legacy models with custom objects are REJECTED unless ALLOW_UNSAFE_MODEL_LOAD=1
    - This prevents arbitrary code execution from untrusted model files
    - See: https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models

    To convert legacy models, run: python tools/convert_legacy_models.py
    """
    # Try PyTorch first
    pt_candidates = sorted(MODELS_DIR.glob("*.pt")) + sorted(MODELS_DIR.glob("*.pth"))
    if pt_candidates:
        try:
            import torch
            path = pt_candidates[0]

            # Security: Load with weights_only=True to prevent arbitrary code execution
            try:
                model = torch.load(path, map_location="cpu", weights_only=True)
            except (pickle.UnpicklingError, RuntimeError, AttributeError) as e:
                # FAIL-CLOSED: Do not load unsafe models by default
                allow_unsafe = os.environ.get("ALLOW_UNSAFE_MODEL_LOAD", "").lower() in ("1", "true", "yes")
                if allow_unsafe:
                    import warnings
                    warnings.warn(
                        f"SECURITY WARNING: Loading model {path} with weights_only=False "
                        f"(ALLOW_UNSAFE_MODEL_LOAD is set). This allows arbitrary code execution. "
                        f"Convert to secure format: python tools/convert_legacy_models.py",
                        SecurityWarning
                    )
                    model = torch.load(path, map_location="cpu", weights_only=False)
                else:
                    raise RuntimeError(
                        f"SECURITY: Model {path} contains non-tensor data and cannot be loaded securely. "
                        f"This is blocked by default to prevent arbitrary code execution. "
                        f"Options: (1) Convert model: python tools/convert_legacy_models.py, "
                        f"(2) Set ALLOW_UNSAFE_MODEL_LOAD=1 if you trust this model's provenance. "
                        f"Original error: {e}"
                    ) from e

            model.eval()
            return ("torch", model, path)
        except Exception:
            pass
    # Try sklearn/joblib
    pkl_candidates = sorted(MODELS_DIR.glob("*.pkl")) + sorted(MODELS_DIR.glob("*.joblib"))
    if pkl_candidates:
        try:
            import joblib
            path = pkl_candidates[0]
            model = joblib.load(path)
            return ("sk", model, path)
        except Exception:
            pass
    raise FileNotFoundError("No supported model found in models/ (.pt/.pth or .pkl/.joblib).")

def _feature_cols(df: pd.DataFrame) -> list:
    # Use standardized features first if present; otherwise originals (except keys)
    prefer_z = [c for c in df.columns if c.endswith("_z")]
    if prefer_z:
        return prefer_z
    exclude = {"timestamp","symbol"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_number_dtype(df[c])]

def _predict(model_kind, model, X):
    if model_kind == "torch":
        import torch
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            y = model(x)
            y_np = y.detach().cpu().numpy()
            if y_np.ndim == 2 and y_np.shape[1] == 1:
                return y_np.ravel()
            if y_np.ndim == 2 and y_np.shape[1] == 2:
                # assume [p0, p1]
                return y_np[:, 1]
            return y_np.ravel()
    else:
        # sklearn-like API
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    return proba[:, 1]
                return proba.max(axis=1)
            except Exception:
                pass
        y = model.predict(X)
        return np.asarray(y).ravel()

def main():
    if not PREPROC.exists():
        raise FileNotFoundError(f"Missing feature pipeline at {PREPROC}. Train first.")
    pipe = FeaturePipeline.load(str(PREPROC))
    model_kind, model, model_path = _load_model()
    print(f"Loaded model: {model_kind} from {model_path.name}")

    files = sorted(PROCESSED.glob("*.feather"))
    if not files:
        raise FileNotFoundError(f"No feather files found in {PROCESSED}.")

    for fp in files:
        sym = fp.stem
        df = pd.read_feather(fp)
        df = pipe.transform_df(df, add_suffix="_z")
        feat_cols = _feature_cols(df)
        X = df[feat_cols].astype(float).to_numpy()
        score = _predict(model_kind, model, X)
        out = pd.DataFrame({
            "timestamp": df["timestamp"].astype("int64"),
            "symbol": df["symbol"].astype(str),
            "close": df["close"].astype(float),
            "score": score.astype(float)
        })
        out = out.dropna().sort_values("timestamp")
        # thresholds from ENV/config (defaults are conservative)
        BUY_THR = get_float("SIGNAL_BUY_THR", 0.6)
        SELL_THR = get_float("SIGNAL_SELL_THR", -0.6)
        DEADZONE = get_float("SIGNAL_DEADZONE", 0.1)

        def _classify(v: float) -> str:
            if abs(v) < DEADZONE:
                return "HOLD"
            if v >= BUY_THR:
                return "BUY"
            if v <= SELL_THR:
                return "SELL"
            return "HOLD"

        # map score -> discrete signal
        out["signal"] = out["score"].astype(float).map(_classify)

        # reorder columns for downstream consumers
        # (price == close; keep score for debugging/analysis)
        out = out[["timestamp","symbol","signal","close","score"]]
        out_path = OUT_DIR / f"{sym}.csv"
        # atomic write
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        out.to_csv(tmp, index=False)
        os.replace(tmp, out_path)

        # explicit log on the last closed hour
        _last = out.iloc[-1]
        print(f"{sym}: last_closed={pd.to_datetime(int(_last['timestamp']), unit='s', utc=True)} "
              f"score={float(_last['score']):.4f} -> signal={_last['signal']} price={float(_last['close']):.6f}")
        print(f"✓ Wrote signals: {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()
