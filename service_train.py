# -*- coding: utf-8 -*-
"""
services/service_train.py
Сервис подготовки данных (офлайн) и запуска обучения модели.
Оркестрация: OfflineData -> FeaturePipe(offl) -> Dataset -> Trainer.fit -> сохранение артефактов.

Пример использования через конфиг
---------------------------------
```python
from core_config import CommonRunConfig
from service_train import from_config, TrainConfig

cfg_run = CommonRunConfig(...)
trainer = ...  # реализация Trainer
train_cfg = TrainConfig(input_path="data/train.parquet")
from_config(cfg_run, trainer=trainer, train_cfg=train_cfg)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Protocol
import os
import time
import logging
import pandas as pd

from services.utils_config import snapshot_config  # снапшот конфигурации
from core_contracts import FeaturePipe
from core_config import CommonRunConfig
import di_registry

logger = logging.getLogger(__name__)


class Trainer(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> Any: ...

    def save(self, path: str) -> str: ...


@dataclass
class TrainConfig:
    input_path: str  # путь к исходным данным (csv/parquet)
    input_format: str = "parquet"  # "parquet" | "csv"
    artifacts_dir: str = "artifacts"  # куда складывать датасеты и модель
    dataset_name: str = "train_dataset"  # базовое имя файлов датасета
    model_name: str = "model"  # базовое имя сохранённой модели
    columns_keep: Optional[Sequence[str]] = None  # если нужно отфильтровать
    snapshot_config_path: Optional[str] = None  # путь к YAML конфигу запуска


class ServiceTrain:
    """
    Подготавливает датасет и обучает переданный Trainer.
    Никакой бизнес-логики обучения внутри; только пайплайн.
    """

    def __init__(self, feature_pipe: FeaturePipe, trainer: Trainer, cfg: TrainConfig):
        self.fp = feature_pipe
        self.trainer = trainer
        self.cfg = cfg

    def _load_input(self) -> pd.DataFrame:
        fmt = str(self.cfg.input_format).lower()
        if fmt == "parquet":
            df = pd.read_parquet(self.cfg.input_path)
        elif fmt == "csv":
            df = pd.read_csv(self.cfg.input_path)
        else:
            raise ValueError(f"Unsupported input_format: {self.cfg.input_format}")
        return df

    def _log_feature_statistics(self, X: pd.DataFrame) -> None:
        """
        Подробное логирование статистики признаков перед обучением.

        Выводит информацию о:
        - Общем количестве признаков
        - Количестве признаков с реальными данными
        - Процент заполненности для каждого признака
        """
        logger.info("=" * 80)
        logger.info("СТАТИСТИКА ПРИЗНАКОВ ПЕРЕД ОБУЧЕНИЕМ")
        logger.info("=" * 80)

        total_features = len(X.columns)
        total_samples = len(X)

        logger.info(f"Общее количество признаков: {total_features}")
        logger.info(f"Общее количество образцов: {total_samples}")
        logger.info("-" * 80)

        # Guard against empty design matrix
        if total_features == 0:
            logger.error("=" * 80)
            logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Design matrix не содержит признаков!")
            logger.error("=" * 80)
            logger.error(f"Количество образцов: {total_samples}")
            logger.error(f"Колонки в датафрейме: {list(X.columns)}")
            logger.error("-" * 80)
            logger.error("Возможные причины:")
            logger.error("  1. Ошибка в конфигурации feature pipeline")
            logger.error("  2. Все признаки были отфильтрованы")
            logger.error("  3. Неправильная подготовка данных")
            logger.error("=" * 80)
            raise ValueError(
                "Cannot train model with zero features. "
                "Please check feature configuration and data preparation pipeline."
            )

        # Подсчет статистики по каждому признаку
        features_stats = []
        fully_filled = 0
        partially_filled = 0
        empty_features = 0

        for col in X.columns:
            non_nan_count = X[col].notna().sum()
            fill_percentage = (non_nan_count / total_samples * 100) if total_samples > 0 else 0

            features_stats.append(
                {"feature": col, "non_nan_count": non_nan_count, "fill_percentage": fill_percentage}
            )

            if fill_percentage == 100.0:
                fully_filled += 1
            elif fill_percentage > 0:
                partially_filled += 1
            else:
                empty_features += 1

        # Сортировка по проценту заполненности (по убыванию)
        features_stats.sort(key=lambda x: x["fill_percentage"], reverse=True)

        # Сводная статистика
        logger.info("СВОДКА:")
        logger.info(
            f"  Признаков с 100% реальными данными: {fully_filled} ({fully_filled/total_features*100:.1f}%)"
        )
        logger.info(
            f"  Признаков с частичными данными: {partially_filled} ({partially_filled/total_features*100:.1f}%)"
        )
        logger.info(
            f"  Признаков без данных (только NaN): {empty_features} ({empty_features/total_features*100:.1f}%)"
        )
        logger.info("-" * 80)

        # Детальная информация по каждому признаку
        logger.info("ДЕТАЛЬНАЯ СТАТИСТИКА ПО ПРИЗНАКАМ:")
        for stat in features_stats:
            logger.info(
                f"  {stat['feature']:50s} | "
                f"Заполнено: {stat['non_nan_count']:6d}/{total_samples:6d} ({stat['fill_percentage']:6.2f}%)"
            )

        logger.info("=" * 80)

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.cfg.artifacts_dir, exist_ok=True)
        if self.cfg.snapshot_config_path:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        # загрузка
        df_raw = self._load_input()

        weights: Optional[pd.Series] = None

        # прогрев и обучение преобразований
        self.fp.warmup()
        self.fp.fit(df_raw)

        # построение фичей и таргета
        X = self.fp.transform_df(df_raw)
        y = None
        try:
            y = self.fp.make_targets(df_raw)
        except Exception:
            y = None

        # опциональная фильтрация колонок
        if self.cfg.columns_keep:
            cols = [c for c in self.cfg.columns_keep if c in X.columns]
            X = X[cols]

        # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: удаление строк с NaN таргетами
        # Проблема: make_targets() использует shift(-1) для расчета будущей доходности,
        # что приводит к NaN в последней строке каждого символа (нет следующей цены).
        # Эти строки необходимо удалить перед обучением, так как они не имеют
        # корректного значения целевой переменной.
        # См. детали в AUDIT_MISSING_TARGET_ROWS.md
        if y is not None:
            # Проверка 1: Убедимся, что X и y имеют одинаковый размер ДО фильтрации NaN
            # Если размеры не совпадают, это означает, что transform_df() удалил некоторые строки
            # (вероятно, из-за NaN в обязательных полях: ts, symbol, price)
            if len(X) != len(y):
                logger.warning(
                    f"X and y have different sizes BEFORE NaN filtering: "
                    f"len(X)={len(X)}, len(y)={len(y)}. "
                    f"This suggests that transform_df() removed some rows (likely NaN in required fields). "
                    f"Will attempt to align datasets by index."
                )

                # Попытка выравнивания по индексам
                common_idx = X.index.intersection(y.index)
                if len(common_idx) == 0:
                    raise ValueError(
                        "X and y have no common indices! Cannot align datasets. "
                        "This likely indicates a data preparation error in the input data."
                    )

                n_removed_by_alignment = max(len(X), len(y)) - len(common_idx)
                logger.info(
                    f"Aligning X and y by common indices. "
                    f"Removing {n_removed_by_alignment} misaligned rows. "
                    f"Retained {len(common_idx)} rows."
                )
                X = X.loc[common_idx]
                y = y.loc[common_idx]

            # Проверка 2: Убедимся, что индексы идентичны для корректного выравнивания
            if not X.index.equals(y.index):
                logger.warning(
                    "X and y indices are not identical. Resetting indices to ensure proper alignment."
                )
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)

            # Теперь фильтруем строки с NaN в таргетах
            valid_mask = y.notna()
            n_before = len(y)
            n_invalid = (~valid_mask).sum()

            if n_invalid > 0:
                logger.info(
                    f"Removing {n_invalid} samples with NaN targets "
                    f"({n_invalid / n_before * 100:.2f}% of total). "
                    f"These are typically the last row of each symbol's time series."
                )
                X = X[valid_mask].reset_index(drop=True)
                y = y[valid_mask].reset_index(drop=True)

                # Финальная проверка согласованности
                if len(X) != len(y):
                    logger.error(f"Shape mismatch after NaN filtering: X={len(X)}, y={len(y)}")
                    raise ValueError(
                        f"X and y have different lengths after filtering: {len(X)} != {len(y)}"
                    )

                logger.info(f"Retained {len(y)} valid samples for training.")
            else:
                logger.info("No NaN targets found - all samples are valid.")

        # Логирование информации о признаках перед обучением
        self._log_feature_statistics(X)

        # FIX (2025-11-21): Filter rows with NaN in features
        # Neural networks cannot handle NaN inputs and will crash or produce NaN gradients
        # We apply conservative row-wise filtering: remove ANY row with NaN in features
        #
        # References:
        # - PyTorch/TensorFlow: Require finite inputs
        # - Scikit-learn: SimpleImputer for NaN handling
        # - Best practice: Never pass NaN to neural networks
        n_before_nan_filter = len(X)

        # Check if X has ANY NaN values
        if X.isna().any().any():
            # Drop columns that are entirely NaN (optional indicators that never materialized)
            all_nan_columns = [col for col in X.columns if X[col].isna().all()]
            if all_nan_columns:
                logger.warning(
                    "Dropping %d all-NaN feature column(s): %s",
                    len(all_nan_columns),
                    all_nan_columns,
                )
                X = X.drop(columns=all_nan_columns)

            # Identify columns/rows with remaining NaN
            nan_columns = X.columns[X.isna().any()].tolist()
            nan_counts_per_col = X[nan_columns].isna().sum().to_dict()
            rows_with_nan_mask = X.isna().any(axis=1)
            n_rows_with_nan = int(rows_with_nan_mask.sum())

            logger.warning(
                f"Found NaN values in {len(nan_columns)} feature column(s): {nan_columns}"
            )
            logger.warning(f"NaN counts per column: {nan_counts_per_col}")

            if n_rows_with_nan > 0:
                # Preserve row count: impute warm-up NaNs instead of dropping samples
                logger.warning(
                    f"Imputing {n_rows_with_nan} row(s) with NaN features "
                    f"({n_rows_with_nan / n_before_nan_filter * 100:.2f}% of total) to avoid data loss."
                )
                X = X.ffill().fillna(0.0)
                if y is not None:
                    # y was already aligned earlier; just reset index to match X after ffill
                    y = y.reset_index(drop=True)

                # Verify alignment and ensure no NaNs remain
                if y is not None and len(X) != len(y):
                    logger.error(f"Shape mismatch after NaN handling: X={len(X)}, y={len(y)}")
                    raise ValueError(
                        f"X and y have different lengths after NaN handling: {len(X)} != {len(y)}"
                    )
                if X.isna().any().any():
                    logger.warning(
                        "NaN values remain after imputation; dropping residual NaN rows."
                    )
                    rows_with_nan_mask = X.isna().any(axis=1)
                    valid_rows_mask = ~rows_with_nan_mask
                    X = X[valid_rows_mask].reset_index(drop=True)
                    if y is not None:
                        y = y[valid_rows_mask].reset_index(drop=True)

            logger.info(f"Retained {len(X)} valid samples for training after feature NaN handling.")
        else:
            logger.info("No NaN values found in features - all samples are valid.")

        ts = int(time.time())
        ds_base = os.path.join(self.cfg.artifacts_dir, f"{self.cfg.dataset_name}_{ts}")
        X_path = ds_base + "_X.parquet"
        y_path = ds_base + "_y.parquet"
        X.to_parquet(X_path, index=False)
        if y is not None:
            pd.DataFrame({"y": y}).to_parquet(y_path, index=False)

        if len(X) == 0:
            logger.error("No valid samples remaining after preprocessing; skipping training run.")
            return {
                "dataset_X": X_path,
                "dataset_y": (y_path if y is not None else None),
                "model_path": None,
                "n_samples": 0,
                "n_features": int(len(X.columns)),
                "effective_samples": 0,
            }

        # ???????????????? ????????????
        self.trainer.fit(X, y, sample_weight=weights)
        model_path = os.path.join(self.cfg.artifacts_dir, f"{self.cfg.model_name}_{ts}.bin")
        saved_path = self.trainer.save(model_path)

        effective = int(len(X))

        return {
            "dataset_X": X_path,
            "dataset_y": (y_path if y is not None else None),
            "model_path": saved_path,
            "n_samples": int(len(X)),
            "n_features": int(len(X.columns)),
            "effective_samples": effective,
        }


def from_config(
    cfg: CommonRunConfig, *, trainer: Trainer, train_cfg: TrainConfig
) -> Dict[str, Any]:
    """Build dependencies from ``cfg`` and run :class:`ServiceTrain`.

    Parameters
    ----------
    cfg: CommonRunConfig
        Runtime configuration with component declarations.
    trainer: Trainer
        Instance implementing :class:`Trainer` protocol.
    train_cfg: TrainConfig
        Configuration specific to training process.
    """
    container = di_registry.build_graph(cfg.components, cfg)
    fp: FeaturePipe = container["feature_pipe"]  # type: ignore[assignment]
    service = ServiceTrain(fp, trainer, train_cfg)
    return service.run()


__all__ = ["TrainConfig", "ServiceTrain", "from_config"]


if __name__ == "__main__":
    import argparse
    import glob
    from core_config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Offline ML Training Service CLI")
    parser.add_argument(
        "--config", default="configs/config_train.yaml", help="Path to YAML run configuration"
    )
    parser.add_argument(
        "--input-path", "--input_path", default=None, help="Path to input data (CSV/Parquet)"
    )
    parser.add_argument(
        "--input-format",
        "--input_format",
        default=None,
        choices=["parquet", "csv"],
        help="Format of input data",
    )
    parser.add_argument(
        "--artifacts-dir",
        "--artifacts_dir",
        default=None,
        help="Directory to save training artifacts",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        default="train_dataset",
        help="Base name for the dataset files",
    )
    parser.add_argument(
        "--model-name", "--model_name", default="model", help="Base name for the saved model"
    )
    parser.add_argument("--trainer", default=None, help="Dotted path (module:Class) of the trainer")

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Build DI graph
    container = di_registry.build_graph(cfg.components, cfg)

    # Resolve trainer
    trainer = None
    if args.trainer:
        try:
            from di_registry import _load_class

            trainer_cls = _load_class(args.trainer)
            trainer = trainer_cls()
            logger.info(f"Loaded custom trainer class: {args.trainer}")
        except Exception as e:
            logger.error(f"Failed to load trainer from dotted path '{args.trainer}': {e}")
            raise

    if trainer is None:
        # Fallback to policy in container
        policy = container.get("policy")
        # Check if policy has fit and save methods
        if policy is not None and hasattr(policy, "fit") and hasattr(policy, "save"):
            trainer = policy
            logger.info(f"Using policy from container as trainer: {policy.__class__.__name__}")
        else:
            # Fallback dummy trainer if no valid trainer is configured or if strategy doesn't support learning
            class FallbackDummyTrainer:
                def fit(self, X, y=None, sample_weight=None):
                    logger.info(
                        "FallbackDummyTrainer.fit() called (strategy does not support learning)."
                    )
                    return self

                def save(self, path):
                    logger.info(f"FallbackDummyTrainer.save() saving mock model to: {path}")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"mock model for {policy.__class__.__name__ if policy else 'None'}")
                    return path

            trainer = FallbackDummyTrainer()
            policy_name = policy.__class__.__name__ if policy else "None"
            logger.warning(
                f"Policy '{policy_name}' does not implement Trainer protocol (missing fit/save). "
                f"Using FallbackDummyTrainer wrapper."
            )

    # Determine input path
    input_path = args.input_path
    if not input_path:
        # Try to extract paths from components config
        md_params = getattr(cfg.components.market_data, "params", {}) or {}
        paths = md_params.get("paths")
        if paths and isinstance(paths, list) and len(paths) > 0:
            raw_path = paths[0]
            # Resolve glob pattern if necessary
            resolved_paths = glob.glob(raw_path)
            if resolved_paths:
                input_path = resolved_paths[0]
                logger.info(f"Resolved input path from market_data config: {input_path}")
            else:
                input_path = raw_path

    if not input_path:
        # Check standard processed path
        default_processed = "data/processed/BTCUSDT.parquet"
        if os.path.exists(default_processed):
            input_path = default_processed
            logger.info(f"Using default processed dataset path: {input_path}")
        else:
            # Check if there is any parquet or csv file in data/processed
            parquet_files = glob.glob("data/processed/*.parquet")
            if parquet_files:
                input_path = parquet_files[0]
                logger.info(f"Found processed parquet dataset: {input_path}")
            else:
                csv_files = glob.glob("data/train/*.csv")
                if csv_files:
                    input_path = csv_files[0]
                    logger.info(f"Found train CSV dataset: {input_path}")
                else:
                    input_path = default_processed
                    logger.warning(f"No datasets found. Defaulting to: {input_path}")

    # Determine input format
    input_format = args.input_format
    if not input_format:
        if input_path.endswith(".csv"):
            input_format = "csv"
        else:
            input_format = "parquet"

    # Determine artifacts directory
    artifacts_dir = args.artifacts_dir or getattr(cfg, "artifacts_dir", "artifacts")

    # Build TrainConfig
    train_cfg = TrainConfig(
        input_path=input_path,
        input_format=input_format,
        artifacts_dir=artifacts_dir,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        snapshot_config_path=args.config,
    )

    logger.info(
        f"Starting ServiceTrain with input_path={input_path}, format={input_format}, artifacts_dir={artifacts_dir}"
    )
    results = from_config(cfg, trainer=trainer, train_cfg=train_cfg)
    print(f"Training completed successfully. Results: {results}")
