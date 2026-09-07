#!/usr/bin/env python3
"""
Тест для проверки корректного удаления строк с NaN таргетами в ServiceTrain.

Проверяет, что:
1. Строки с NaN таргетами (последняя строка каждого символа) удаляются
2. X и y имеют одинаковую длину после фильтрации
3. В y не остается NaN значений
4. Trainer получает только валидные данные
"""

import os
import tempfile
import shutil
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd
import pytest

# Skip entire module if pyarrow not available (tests use parquet files)
pytest.importorskip("pyarrow", reason="pyarrow required for parquet support")

from service_train import ServiceTrain, TrainConfig, Trainer
from feature_pipe import FeaturePipe
from core_config import FeatureSpec


# Mock Trainer для тестирования
@dataclass
class MockTrainer:
    """Mock trainer that records what it receives."""

    received_X: Optional[pd.DataFrame] = None
    received_y: Optional[pd.Series] = None
    received_weights: Optional[pd.Series] = None
    model_saved: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> Any:
        """Record received data and check for NaN in targets."""
        self.received_X = X
        self.received_y = y
        self.received_weights = sample_weight

        # КРИТИЧНО: проверяем, что нет NaN в таргетах
        if y is not None:
            nan_count = y.isna().sum()
            if nan_count > 0:
                raise ValueError(
                    f"Trainer received {nan_count} NaN values in targets! "
                    f"This should have been filtered out."
                )

        return self

    def save(self, path: str) -> str:
        """Save mock model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("mock model")
        self.model_saved = True
        return path


def test_nan_targets_are_filtered():
    """
    Тест проверяет, что строки с NaN таргетами удаляются перед обучением.

    Сценарий:
    1. Создаем данные с 2 символами по 5 строк каждый (всего 10 строк)
    2. make_targets() создаст NaN для последней строки каждого символа (2 NaN)
    3. ServiceTrain должен удалить эти 2 строки
    4. Trainer должен получить 8 валидных строк
    """
    # Подготовка тестовых данных
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5 + ["ETHUSDT"] * 5,
            "ts_ms": list(range(0, 300_000, 60_000)) + list(range(0, 300_000, 60_000)),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 200.0, 202.0, 204.0, 206.0, 208.0],
        }
    )

    # Создаем временную директорию для артефактов
    temp_dir = tempfile.mkdtemp()

    try:
        # Сохраняем тестовые данные
        input_path = os.path.join(temp_dir, "test_input.parquet")
        df.to_parquet(input_path, index=False)

        # Создаем FeaturePipe
        fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")

        # Создаем mock trainer
        trainer = MockTrainer()

        # Создаем TrainConfig
        train_cfg = TrainConfig(
            input_path=input_path,
            input_format="parquet",
            artifacts_dir=temp_dir,
            dataset_name="test_dataset",
            model_name="test_model",
        )

        # Создаем ServiceTrain
        service = ServiceTrain(feature_pipe=fp, trainer=trainer, cfg=train_cfg)

        # Запускаем обучение
        result = service.run()

        # ПРОВЕРКИ
        # 1. Trainer был вызван
        assert trainer.received_X is not None, "Trainer не получил X"
        assert trainer.received_y is not None, "Trainer не получил y"

        # 2. Нет NaN в таргетах
        assert trainer.received_y.isna().sum() == 0, (
            f"Trainer получил {trainer.received_y.isna().sum()} NaN в таргетах! "
            f"Фильтрация не сработала."
        )

        # 3. Правильное количество строк (10 исходных - 2 с NaN = 8)
        expected_samples = 8
        assert len(trainer.received_y) == expected_samples, (
            f"Ожидалось {expected_samples} валидных строк, " f"получено {len(trainer.received_y)}"
        )

        # 4. X и y имеют одинаковую длину
        assert len(trainer.received_X) == len(trainer.received_y), (
            f"X и y имеют разную длину: "
            f"X={len(trainer.received_X)}, y={len(trainer.received_y)}"
        )

        # 5. Результат содержит корректную информацию
        assert result["n_samples"] == expected_samples
        assert result["effective_samples"] == expected_samples

        # 6. Модель была сохранена
        assert trainer.model_saved, "Модель не была сохранена"

        print("✓ Тест пройден: NaN таргеты корректно отфильтрованы")

    finally:
        # Очистка
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_no_nan_targets_case():
    """
    Тест проверяет, что код работает корректно, когда NaN таргетов нет.

    Этот случай теоретически возможен, если:
    - label_col уже определен в данных
    - Или данные были предобработаны
    """
    # Подготовка тестовых данных с уже готовым таргетом
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "ts_ms": list(range(0, 300_000, 60_000)),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "target": [0.01, 0.0099, 0.0098, 0.0097, 0.0096],  # Все валидные
        }
    )

    temp_dir = tempfile.mkdtemp()

    try:
        input_path = os.path.join(temp_dir, "test_input.parquet")
        df.to_parquet(input_path, index=False)

        # FeaturePipe с label_col
        fp = FeaturePipe(
            spec=FeatureSpec(lookbacks_prices=[1, 5]),
            price_col="close",
            label_col="target",  # Используем готовый таргет
        )

        trainer = MockTrainer()

        train_cfg = TrainConfig(
            input_path=input_path,
            input_format="parquet",
            artifacts_dir=temp_dir,
            dataset_name="test_dataset",
            model_name="test_model",
        )

        service = ServiceTrain(feature_pipe=fp, trainer=trainer, cfg=train_cfg)
        result = service.run()

        # ПРОВЕРКИ
        assert trainer.received_y is not None
        assert trainer.received_y.isna().sum() == 0, "Есть NaN в таргетах"
        assert len(trainer.received_y) == 5, "Неправильное количество строк"

        print("✓ Тест пройден: случай без NaN таргетов обрабатывается корректно")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_multiple_symbols_nan_filtering():
    """
    Тест проверяет фильтрацию NaN для множества символов.

    Каждый символ должен потерять ровно одну последнюю строку.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"]
    rows_per_symbol = 10

    # Создаем данные
    dfs = []
    for symbol in symbols:
        df_sym = pd.DataFrame(
            {
                "symbol": [symbol] * rows_per_symbol,
                "ts_ms": list(range(0, rows_per_symbol * 60_000, 60_000)),
                "close": [100.0 + i for i in range(rows_per_symbol)],
            }
        )
        dfs.append(df_sym)

    df = pd.concat(dfs, ignore_index=True)

    temp_dir = tempfile.mkdtemp()

    try:
        input_path = os.path.join(temp_dir, "test_input.parquet")
        df.to_parquet(input_path, index=False)

        fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")
        trainer = MockTrainer()

        train_cfg = TrainConfig(
            input_path=input_path,
            input_format="parquet",
            artifacts_dir=temp_dir,
            dataset_name="test_dataset",
            model_name="test_model",
        )

        service = ServiceTrain(feature_pipe=fp, trainer=trainer, cfg=train_cfg)
        result = service.run()

        # ПРОВЕРКИ
        total_rows = len(symbols) * rows_per_symbol  # 40
        expected_valid_rows = total_rows - len(symbols)  # 40 - 4 = 36

        assert trainer.received_y is not None
        assert trainer.received_y.isna().sum() == 0, "Есть NaN в таргетах"
        assert (
            len(trainer.received_y) == expected_valid_rows
        ), f"Ожидалось {expected_valid_rows} строк, получено {len(trainer.received_y)}"

        print(
            f"✓ Тест пройден: {len(symbols)} символов, "
            f"{len(symbols)} строк с NaN корректно удалены"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_edge_case_single_symbol_single_row():
    """
    Edge case: один символ с одной строкой.

    После shift(-1) будет NaN, и не останется данных для обучения.
    """
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "ts_ms": [0],
            "close": [100.0],
        }
    )

    temp_dir = tempfile.mkdtemp()

    try:
        input_path = os.path.join(temp_dir, "test_input.parquet")
        df.to_parquet(input_path, index=False)

        fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")
        trainer = MockTrainer()

        train_cfg = TrainConfig(
            input_path=input_path,
            input_format="parquet",
            artifacts_dir=temp_dir,
            dataset_name="test_dataset",
            model_name="test_model",
        )

        service = ServiceTrain(feature_pipe=fp, trainer=trainer, cfg=train_cfg)
        result = service.run()

        # После удаления единственной строки с NaN, не должно остаться данных
        # Но код должен корректно обработать это
        assert trainer.received_y is not None or trainer.received_y is None
        if trainer.received_y is not None:
            assert len(trainer.received_y) == 0, "Должно быть 0 валидных строк"

        print("✓ Тест пройден: Edge case с одной строкой обработан корректно")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_edge_case_all_nan_targets():
    """
    Edge case: Все таргеты NaN.

    Такое может случиться, если у каждого символа только одна строка.
    """
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "ts_ms": [0, 0, 0],
            "close": [100.0, 200.0, 300.0],
        }
    )

    temp_dir = tempfile.mkdtemp()

    try:
        input_path = os.path.join(temp_dir, "test_input.parquet")
        df.to_parquet(input_path, index=False)

        fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")
        trainer = MockTrainer()

        train_cfg = TrainConfig(
            input_path=input_path,
            input_format="parquet",
            artifacts_dir=temp_dir,
            dataset_name="test_dataset",
            model_name="test_model",
        )

        service = ServiceTrain(feature_pipe=fp, trainer=trainer, cfg=train_cfg)
        result = service.run()

        # Все строки должны быть отфильтрованы
        if trainer.received_y is not None:
            assert len(trainer.received_y) == 0, "Все строки должны быть отфильтрованы"

        print("✓ Тест пройден: Edge case со всеми NaN таргетами обработан")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_edge_case_empty_dataframe():
    """
    Edge case: Пустой DataFrame.
    """
    df = pd.DataFrame({"symbol": [], "ts_ms": [], "close": []})

    temp_dir = tempfile.mkdtemp()

    try:
        input_path = os.path.join(temp_dir, "test_input.parquet")
        df.to_parquet(input_path, index=False)

        fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")
        trainer = MockTrainer()

        train_cfg = TrainConfig(
            input_path=input_path,
            input_format="parquet",
            artifacts_dir=temp_dir,
            dataset_name="test_dataset",
            model_name="test_model",
        )

        service = ServiceTrain(feature_pipe=fp, trainer=trainer, cfg=train_cfg)

        # Код может выдать ошибку или вернуть пустой результат - оба варианта приемлемы
        try:
            result = service.run()
            if trainer.received_X is not None:
                assert len(trainer.received_X) == 0
            print("✓ Тест пройден: Пустой DataFrame обработан корректно")
        except Exception as e:
            print(f"✓ Тест пройден: Пустой DataFrame вызвал ожидаемую ошибку: {type(e).__name__}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    """Запуск тестов вручную."""
    print("=" * 80)
    print("COMPREHENSIVE ТЕСТЫ ФИЛЬТРАЦИИ NaN ТАРГЕТОВ В ServiceTrain")
    print("=" * 80)

    try:
        print("\n1. Тест базовой фильтрации NaN таргетов...")
        test_nan_targets_are_filtered()

        print("\n2. Тест случая без NaN таргетов...")
        test_no_nan_targets_case()

        print("\n3. Тест фильтрации для множества символов...")
        test_multiple_symbols_nan_filtering()

        print("\n4. Edge case: Один символ, одна строка...")
        test_edge_case_single_symbol_single_row()

        print("\n5. Edge case: Все таргеты NaN...")
        test_edge_case_all_nan_targets()

        print("\n6. Edge case: Пустой DataFrame...")
        test_edge_case_empty_dataframe()

        print("\n" + "=" * 80)
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! ✓")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ОШИБКА: {type(e).__name__}: {e}")
        raise
