#!/usr/bin/env python3
"""
Utility functions for analyzing and improving normalization.

Optional enhancements to the current tanh-based normalization approach.
Use only if you observe specific issues with the current implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_feature_distributions(
    data: pd.DataFrame,
    feature_columns: List[str],
    output_dir: Path = Path("artifacts/normalization_analysis"),
) -> Dict[str, Dict[str, float]]:
    """
    Analyze raw feature distributions to identify potential normalization issues.

    Returns statistics for each feature:
    - mean, median, std, min, max
    - percentiles (1, 5, 25, 75, 95, 99)
    - skewness, kurtosis
    - % of extreme values (beyond ±3σ)

    Use this to:
    1. Identify heavy-tailed features (high kurtosis)
    2. Find features with extreme outliers
    3. Determine if adaptive scaling would help
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for col in feature_columns:
        if col not in data.columns:
            continue

        values = pd.to_numeric(data[col], errors="coerce").dropna()

        if len(values) == 0:
            continue

        # Basic statistics
        col_stats = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(values.quantile(0.01)),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "skew": float(values.skew()),
            "kurtosis": float(values.kurtosis()),
        }

        # Calculate % of extreme values (beyond ±3σ)
        mean = col_stats["mean"]
        std = col_stats["std"]
        if std > 0:
            extreme_pct = (np.abs(values - mean) > 3 * std).mean() * 100
            col_stats["extreme_pct"] = float(extreme_pct)
        else:
            col_stats["extreme_pct"] = 0.0

        # After tanh normalization, how much saturation?
        normalized = np.tanh(values)
        saturated_pct = (np.abs(normalized) > 0.95).mean() * 100
        col_stats["tanh_saturation_pct"] = float(saturated_pct)

        stats[col] = col_stats

        # Warn about potential issues
        if col_stats["tanh_saturation_pct"] > 10:
            print(f"⚠️ {col}: {col_stats['tanh_saturation_pct']:.1f}% of values saturate tanh")
            print(f"   Consider adaptive scaling if this feature is important")

        if col_stats["kurtosis"] > 10:
            print(f"⚠️ {col}: Heavy-tailed distribution (kurtosis={col_stats['kurtosis']:.1f})")

    # Save to JSON
    import json

    with open(output_dir / "feature_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def plot_normalization_comparison(
    raw_values: np.ndarray,
    feature_name: str = "Feature",
    output_path: Path = Path("artifacts/normalization_comparison.png"),
) -> None:
    """
    Visualize how different normalization strategies affect a feature.

    Compares:
    1. Raw values
    2. tanh(raw)
    3. z-score
    4. tanh with adaptive scaling

    Use this to understand if current normalization is appropriate.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Raw values
    axes[0, 0].hist(raw_values, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title(f"Raw {feature_name}")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Current approach: tanh
    tanh_normalized = np.tanh(raw_values)
    axes[0, 1].hist(tanh_normalized, bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[0, 1].set_title(f"tanh({feature_name}) - CURRENT")
    axes[0, 1].set_xlabel("Normalized Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(x=-1, color="red", linestyle="--", alpha=0.5, label="Bounds")
    axes[0, 1].axvline(x=1, color="red", linestyle="--", alpha=0.5)
    axes[0, 1].legend()

    # 3. Alternative: z-score (for comparison)
    mean = raw_values.mean()
    std = raw_values.std()
    if std > 0:
        z_score = (raw_values - mean) / std
        axes[1, 0].hist(z_score, bins=50, edgecolor="black", alpha=0.7, color="orange")
        axes[1, 0].set_title(f"z-score({feature_name}) - Unbounded")
        axes[1, 0].set_xlabel("Normalized Value")
        axes[1, 0].set_ylabel("Frequency")

    # 4. Adaptive tanh (using 95th percentile)
    scale = np.percentile(np.abs(raw_values), 95)
    if scale > 0:
        adaptive_tanh = np.tanh(raw_values / scale)
        axes[1, 1].hist(adaptive_tanh, bins=50, edgecolor="black", alpha=0.7, color="purple")
        axes[1, 1].set_title(f"tanh({feature_name} / p95) - Adaptive")
        axes[1, 1].set_xlabel("Normalized Value")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].axvline(x=-1, color="red", linestyle="--", alpha=0.5)
        axes[1, 1].axvline(x=1, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def suggest_adaptive_scales(
    data: pd.DataFrame, feature_columns: List[str], percentile: float = 95.0
) -> Dict[str, float]:
    """
    Calculate adaptive scaling factors for features that saturate tanh.

    Returns a dict of {feature_name: scale_factor} where:
        scale_factor = percentile(abs(feature_values))

    Usage:
        scales = suggest_adaptive_scales(train_data, features)
        # Then in obs_builder.pyx:
        # normalized = tanh(value / scales[feature_name])
    """
    scales = {}

    for col in feature_columns:
        if col not in data.columns:
            continue

        values = pd.to_numeric(data[col], errors="coerce").dropna()
        if len(values) == 0:
            continue

        # Calculate scale as percentile of absolute values
        scale = np.percentile(np.abs(values), percentile)

        # Check if adaptive scaling would help
        current_saturation = (np.abs(np.tanh(values)) > 0.95).mean() * 100
        adaptive_saturation = (np.abs(np.tanh(values / max(scale, 1.0))) > 0.95).mean() * 100

        if adaptive_saturation < current_saturation * 0.5:  # 50% improvement
            scales[col] = float(max(scale, 1.0))
            print(f"✓ {col}: Adaptive scaling recommended")
            print(f"  Saturation: {current_saturation:.1f}% → {adaptive_saturation:.1f}%")
            print(f"  Scale factor: {scale:.4f}")

    return scales


def validate_normalization_consistency(
    train_obs: np.ndarray, val_obs: np.ndarray, feature_names: List[str] = None, alpha: float = 0.01
) -> Dict[str, bool]:
    """
    Validate that training and validation observations have consistent distributions.

    Uses Kolmogorov-Smirnov test to detect distribution shifts.

    Returns:
        Dict mapping feature names to boolean (True if consistent, False if shifted)

    Interpretation:
        - False: Significant distribution shift detected (p < alpha)
          May indicate: data leakage, inconsistent normalization, or natural shift
    """
    from scipy.stats import ks_2samp

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(train_obs.shape[1])]

    results = {}

    print("\n=== Normalization Consistency Validation ===")
    print(f"Alpha level: {alpha}")
    print(f"Train samples: {len(train_obs)}, Val samples: {len(val_obs)}")

    for idx, name in enumerate(feature_names[: train_obs.shape[1]]):
        stat, p_value = ks_2samp(train_obs[:, idx], val_obs[:, idx])

        is_consistent = p_value >= alpha
        results[name] = is_consistent

        status = "✓" if is_consistent else "⚠️"
        print(f"{status} {name:30s}: KS={stat:.4f}, p={p_value:.4e}")

        if not is_consistent:
            # Additional diagnostics
            train_mean = train_obs[:, idx].mean()
            val_mean = val_obs[:, idx].mean()
            train_std = train_obs[:, idx].std()
            val_std = val_obs[:, idx].std()

            print(f"   Train: mean={train_mean:.4f}, std={train_std:.4f}")
            print(f"   Val:   mean={val_mean:.4f}, std={val_std:.4f}")

    consistent_count = sum(results.values())
    total_count = len(results)
    print(
        f"\nConsistent features: {consistent_count}/{total_count} "
        f"({consistent_count/total_count*100:.1f}%)"
    )

    return results


def compute_adaptive_scaling_params(
    train_data: pd.DataFrame,
    feature_columns: List[str],
    percentile: float = 95.0,
    output_path: Path = Path("models/adaptive_scales.json"),
) -> Dict[str, float]:
    """
    Compute and save adaptive scaling parameters from training data.

    This is OPTIONAL enhancement - only use if you observe that:
    1. Important features saturate tanh (>10% of values → ±1)
    2. Model fails to distinguish between extreme values
    3. Performance improves with adaptive scaling

    Usage in obs_builder.pyx:
        # Load scales at initialization
        with open("models/adaptive_scales.json") as f:
            scales = json.load(f)

        # Apply during normalization
        scale = scales.get(feature_name, 1.0)
        normalized = tanh(value / scale)
    """
    scales = suggest_adaptive_scales(train_data, feature_columns, percentile)

    if scales:
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(scales, f, indent=2)
        print(f"\n✓ Saved {len(scales)} adaptive scales to {output_path}")
    else:
        print("\n✓ No adaptive scaling needed - current tanh normalization is optimal")

    return scales


# Example usage
if __name__ == "__main__":
    print(
        """
    Normalization Analysis Utilities
    =================================

    This module provides tools to analyze and optionally improve normalization.

    Current approach (tanh) is already optimal for most cases!
    Only use these tools if you observe specific issues.

    Example workflow:
    -----------------
    1. Analyze feature distributions:
       stats = analyze_feature_distributions(train_df, feature_cols)

    2. Visualize specific features:
       plot_normalization_comparison(train_df['garch_14d'].values, 'GARCH 14d')

    3. Check for saturation issues:
       scales = suggest_adaptive_scales(train_df, feature_cols)

    4. Validate train/val consistency:
       validate_normalization_consistency(train_obs, val_obs, feature_names)

    Remember: Don't optimize prematurely! The current approach works well.
    """
    )
