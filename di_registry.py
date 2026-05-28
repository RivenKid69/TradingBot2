# -*- coding: utf-8 -*-
"""
di_registry.py
Простой DI-контейнер для сборки компонентов по dotted path "module:Class".
Поддерживает:
  - указание параметров конструктора в конфиге (params)
  - авто-подстановку зависимостей по имени параметра конструктора,
    если такой компонент уже собран в контейнере (name → instance)
  - Exchange adapters для multi-exchange support (Binance, Alpaca)
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Dict, Mapping, Optional, Type, TypeVar, get_type_hints

from core_errors import ConfigError
from core_config import (
    ComponentSpec,
    Components,
    CommonRunConfig,
    RetryConfig,
    AdvRuntimeConfig,
    PortfolioConfig,
    SpotCostConfig,
    ExecutionRuntimeConfig,
)
from impl_quantizer import QuantizerImpl


# =============================================================================
# Exchange Adapter Support
# =============================================================================

def _build_exchange_adapters(
    exchange_config: Mapping[str, Any],
    container: Dict[Any, Any],
) -> None:
    """
    Build exchange adapters from configuration and add to container.

    Args:
        exchange_config: Exchange configuration dict with vendor/settings
        container: DI container to add adapters to
    """
    try:
        from adapters.config import ExchangeConfig
        from adapters.base import (
            MarketDataAdapter,
            FeeAdapter,
            TradingHoursAdapter,
            ExchangeInfoAdapter,
            OrderExecutionAdapter,
        )

        # Parse exchange config
        if isinstance(exchange_config, ExchangeConfig):
            cfg = exchange_config
        else:
            cfg = ExchangeConfig.from_dict(exchange_config)

        container["exchange_config"] = cfg
        container["exchange_vendor"] = cfg.vendor

        # Create and register adapters
        try:
            market_data_adapter = cfg.create_market_data_adapter()
            container["market_data_adapter"] = market_data_adapter
            container[MarketDataAdapter] = market_data_adapter
        except Exception as e:
            logger.debug(f"Could not create market data adapter: {e}")

        try:
            fee_adapter = cfg.create_fee_adapter()
            container["fee_adapter"] = fee_adapter
            container[FeeAdapter] = fee_adapter
        except Exception as e:
            logger.debug(f"Could not create fee adapter: {e}")

        try:
            trading_hours_adapter = cfg.create_trading_hours_adapter()
            container["trading_hours_adapter"] = trading_hours_adapter
            container[TradingHoursAdapter] = trading_hours_adapter
        except Exception as e:
            logger.debug(f"Could not create trading hours adapter: {e}")

        try:
            exchange_info_adapter = cfg.create_exchange_info_adapter()
            container["exchange_info_adapter"] = exchange_info_adapter
            container[ExchangeInfoAdapter] = exchange_info_adapter
        except Exception as e:
            logger.debug(f"Could not create exchange info adapter: {e}")

        logger.info(f"Registered exchange adapters for vendor: {cfg.vendor}")

    except ImportError as e:
        logger.warning(f"Exchange adapters not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to build exchange adapters: {e}")


def _load_class(dotted: str):
    try:
        module_name, cls_name = dotted.split(":")
    except ValueError as e:
        raise ConfigError(f'Некорректный dotted path "{dotted}". Ожидается "module.submodule:ClassName"') from e
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, cls_name)
    except AttributeError as e:
        raise ConfigError(f'В модуле "{module_name}" нет класса "{cls_name}"') from e
    return cls


T = TypeVar("T")


logger = logging.getLogger(__name__)


def _instantiate(target_cls, params: Dict[str, Any], container: Mapping[Any, Any]) -> Any:
    """
    Создание экземпляра с учётом DI:
      - сопоставляем сигнатуру конструктора аргументам
      - если какое-то имя аргумента совпадает с уже созданным компонентом — подставляем его
      - при конфликте приоритет у явного params
    """
    sig = inspect.signature(target_cls.__init__)
    try:
        hints = get_type_hints(target_cls.__init__)
    except Exception:
        hints = {}
    kwargs: Dict[str, Any] = {}
    
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if name in params:
            kwargs[name] = params[name]
        elif name in container:
            kwargs[name] = container[name]
        else:
            ann = hints.get(name)
            if ann in container:
                kwargs[name] = container[ann]
            else:
                if p.default is not inspect._empty or p.kind in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                ):
                    continue
                # оставляем незаполненным — конструктор может это принять
                
    if has_var_keyword:
        for k, v in params.items():
            if k not in kwargs and k not in sig.parameters:
                kwargs[k] = v
                
    return target_cls(**kwargs)


def build_component(name: str, spec: ComponentSpec, container: Dict[Any, Any]) -> Any:
    cls = _load_class(spec.target)
    instance = _instantiate(cls, spec.params or {}, container)
    container[name] = instance
    for base in inspect.getmro(cls):
        if base is object or base.__module__.startswith("typing"):
            continue
        container[base] = instance
    return instance


_GLOBAL_CONTAINER: Dict[Any, Any] | None = None


def resolve(key: Type[T], container: Mapping[Any, Any] | None = None) -> T:
    cont = container if container is not None else _GLOBAL_CONTAINER
    if cont is None or key not in cont:
        raise KeyError(key)
    return cont[key]


def build_graph(components: Components, run_config: Optional[CommonRunConfig] = None) -> Dict[Any, Any]:
    """
    Сборка графа в последовательности: market_data → feature_pipe → policy → risk_guards → executor → backtest_engine
    (BacktestEngine опционален.)

    Also builds exchange adapters if 'exchange' config is present.
    """
    container: Dict[Any, Any] = {}
    if run_config is not None:
        container["run_config"] = run_config
        container["retry_cfg"] = run_config.retry
        container[RetryConfig] = run_config.retry
        portfolio_cfg = getattr(run_config, "portfolio", None)
        if isinstance(portfolio_cfg, PortfolioConfig):
            container["portfolio"] = portfolio_cfg
            container[PortfolioConfig] = portfolio_cfg
        costs_cfg = getattr(run_config, "costs", None)
        if isinstance(costs_cfg, SpotCostConfig):
            container["costs"] = costs_cfg
            container[SpotCostConfig] = costs_cfg
        exec_cfg = getattr(run_config, "execution", None)
        if isinstance(exec_cfg, ExecutionRuntimeConfig):
            container["execution"] = exec_cfg
            container["execution_config"] = exec_cfg
            container[ExecutionRuntimeConfig] = exec_cfg
        q_cfg = getattr(run_config, "quantizer", None)
        if isinstance(q_cfg, Mapping):
            try:
                quantizer = QuantizerImpl.from_dict(dict(q_cfg))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to build QuantizerImpl from config: %s", exc)
            else:
                container["quantizer"] = quantizer
                container[QuantizerImpl] = quantizer
        adv_cfg = getattr(run_config, "adv", None)
        if isinstance(adv_cfg, AdvRuntimeConfig):
            container["adv_runtime_config"] = adv_cfg
            container[AdvRuntimeConfig] = adv_cfg
            container.setdefault("adv", adv_cfg)

        # Build exchange adapters if exchange config is present
        exchange_cfg = getattr(run_config, "exchange", None)
        if exchange_cfg is not None and isinstance(exchange_cfg, Mapping):
            _build_exchange_adapters(exchange_cfg, container)

    build_component("market_data", components.market_data, container)
    build_component("feature_pipe", components.feature_pipe, container)
    build_component("policy", components.policy, container)
    build_component("risk_guards", components.risk_guards, container)

    executor_spec = components.executor
    if run_config is not None:
        exec_cfg = getattr(run_config, "execution", None)
        mode_value = getattr(exec_cfg, "mode", None) if exec_cfg is not None else None
        if isinstance(mode_value, str) and mode_value.lower() == "bar":
            target = str(executor_spec.target or "")
            if target == "impl_sim_executor:SimExecutor":
                executor_spec = ComponentSpec.parse_obj(
                    {
                        "target": "impl_bar_executor:BarExecutor",
                        "params": dict(executor_spec.params or {}),
                    }
                )

    build_component("executor", executor_spec, container)
    if "executor" in container:
        sim_obj = container["executor"]
        if type(sim_obj).__name__ != "ExecutionSimulator":
            if hasattr(sim_obj, "_sim"):
                sim_obj = getattr(sim_obj, "_sim")
            elif hasattr(sim_obj, "sim"):
                sim_obj = getattr(sim_obj, "sim")
        container["sim"] = sim_obj
    if "run_config" in container:
        container["cfg"] = container["run_config"]
    if components.backtest_engine:
        build_component("backtest_engine", components.backtest_engine, container)
    global _GLOBAL_CONTAINER
    _GLOBAL_CONTAINER = container
    return container
