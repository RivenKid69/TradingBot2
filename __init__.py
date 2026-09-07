# __init__.py
# Note: This file uses relative imports which require a parent package context.
# When running pytest from project root, this file may be imported directly,
# so we wrap imports in try/except to prevent test collection failures.

try:
    from .quantizer import Quantizer, load_filters, SymbolFilters
    from .fees import FeesModel, FundingCalculator, FundingEvent
    from .slippage import (
        SlippageConfig,
        estimate_slippage_bps,
        apply_slippage_price,
        compute_spread_bps_from_quotes,
        mid_from_quotes,
    )
    from .execution_algos import (
        BaseExecutor,
        MarketChild,
        TakerExecutor,
        TWAPExecutor,
        POVExecutor,
        MidOffsetLimitExecutor,
    )

    try:
        from .latency import LatencyModel
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        LatencyModel = None  # type: ignore[assignment]
    from .risk import RiskManager, RiskConfig, RiskEvent

    try:
        from .sim_logging import LogWriter, LogConfig
    except Exception:  # pragma: no cover - optional dependency
        LogWriter = None  # type: ignore
        LogConfig = None  # type: ignore

    __all__ = [
        "Quantizer",
        "load_filters",
        "SymbolFilters",
        "FeesModel",
        "FundingCalculator",
        "FundingEvent",
        "SlippageConfig",
        "estimate_slippage_bps",
        "apply_slippage_price",
        "compute_spread_bps_from_quotes",
        "mid_from_quotes",
        "BaseExecutor",
        "MarketChild",
        "TakerExecutor",
        "TWAPExecutor",
        "POVExecutor",
        "MidOffsetLimitExecutor",
        "LatencyModel",
        "RiskManager",
        "RiskConfig",
        "RiskEvent",
        "LogWriter",
        "LogConfig",
    ]

except ImportError:
    # When imported directly (not as a package), skip relative imports.
    # This allows pytest to scan the project without failing.
    __all__ = []
