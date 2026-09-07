# -*- coding: utf-8 -*-
"""
core_errors.py
Общие исключения для системы.
"""


class BotError(Exception):
    """Базовая ошибка системы."""


class ConfigError(BotError):
    """Ошибка конфигурации/валидации."""


class DataError(BotError):
    """Ошибка данных/источника."""


class ExecutionError(BotError):
    """Ошибка исполнения ордеров."""


class RiskViolation(BotError):
    """Нарушение риск-ограничений."""


class QuantizeError(BotError):
    """Ошибка квантизации цен/количеств."""


class BacktestError(BotError):
    """Ошибка работы бэктест-движка."""


# =============================================================================
# Options-Specific Errors (Phase 1)
# =============================================================================


class OptionsError(BotError):
    """Base options error class."""


class GreeksCalculationError(OptionsError):
    """
    Greeks calculation failure.

    Raised when Greeks calculation fails due to invalid inputs,
    numerical instability, or convergence issues.
    """


class IVConvergenceError(OptionsError):
    """
    Implied volatility solver failed to converge.

    Raised when the IV solver (Newton-Raphson, Brent, or hybrid)
    cannot find a solution within the specified tolerance and iterations.
    """

    def __init__(
        self,
        message: str = "IV solver failed to converge",
        iterations: int = 0,
        last_estimate: float = 0.0,
        target_price: float = 0.0,
        model_price: float = 0.0,
    ):
        super().__init__(message)
        self.iterations = iterations
        self.last_estimate = last_estimate
        self.target_price = target_price
        self.model_price = model_price


class PricingError(OptionsError):
    """
    Options pricing failure.

    Raised when pricing model fails due to invalid inputs
    or numerical issues.
    """


class ExerciseError(OptionsError):
    """
    Exercise or assignment error.

    Raised when early exercise analysis fails or when
    exercise/assignment processing encounters issues.
    """


class MarginError(OptionsError):
    """
    OCC margin calculation error.

    Raised when margin requirement calculation fails
    due to missing data or invalid position.
    """


class CalibrationError(OptionsError):
    """
    Model calibration failure.

    Raised when calibration of volatility surface,
    jump parameters, or other model parameters fails.
    """
