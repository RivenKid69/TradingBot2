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
    ) -> Any:
        ...

    def save(self, path: str) -> str:
        ...


@dataclass
class TrainConfig:
    input_path: str                       # путь к исходным данным (csv/parquet)
    input_format: str = "parquet"         # "parquet" | "csv"
    artifacts_dir: str = "artifacts"      # куда складывать датасеты и модель
    dataset_name: str = "train_dataset"   # базовое имя файлов датасета
    model_name: str = "model"             # базовое имя сохранённой модели
    columns_keep: Optional[Sequence[str]] = None  # если нужно отфильтровать
    snapshot_config_path: Optional[str] = None    # путь к YAML конфигу запуска


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

        # Подсчет статистики по каждому признаку
        features_stats = []
        fully_filled = 0
        partially_filled = 0
        empty_features = 0

        for col in X.columns:
            non_nan_count = X[col].notna().sum()
            fill_percentage = (non_nan_count / total_samples * 100) if total_samples > 0 else 0

            features_stats.append({
                'feature': col,
                'non_nan_count': non_nan_count,
                'fill_percentage': fill_percentage
            })

            if fill_percentage == 100.0:
                fully_filled += 1
            elif fill_percentage > 0:
                partially_filled += 1
            else:
                empty_features += 1

        # Сортировка по проценту заполненности (по убыванию)
        features_stats.sort(key=lambda x: x['fill_percentage'], reverse=True)

        # Сводная статистика
        logger.info("СВОДКА:")
        logger.info(f"  Признаков с 100% реальными данными: {fully_filled} ({fully_filled/total_features*100:.1f}%)")
        logger.info(f"  Признаков с частичными данными: {partially_filled} ({partially_filled/total_features*100:.1f}%)")
        logger.info(f"  Признаков без данных (только NaN): {empty_features} ({empty_features/total_features*100:.1f}%)")
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

        # Логирование информации о признаках перед обучением
        self._log_feature_statistics(X)

        # сохранение датасета
        ts = int(time.time())
        ds_base = os.path.join(self.cfg.artifacts_dir, f"{self.cfg.dataset_name}_{ts}")
        X_path = ds_base + "_X.parquet"
        y_path = ds_base + "_y.parquet"
        X.to_parquet(X_path, index=False)
        if y is not None:
            pd.DataFrame({"y": y}).to_parquet(y_path, index=False)

        # обучение модели
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


def from_config(cfg: CommonRunConfig, *, trainer: Trainer, train_cfg: TrainConfig) -> Dict[str, Any]:
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
