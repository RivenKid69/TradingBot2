# strategies/custom_options.py
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class OptionsDeltaNeutralStrategy(BaseSignalPolicy):
    """
    Delta Neutral Dynamic Hedging Strategy for Options.
    Hedges portfolio delta limit by buying/selling the underlying asset.
    """
    required_features = ("ref_price", "portfolio_delta")

    def __init__(self) -> None:
        super().__init__()
        self.hedge_threshold = 0.15
        self.tif = TimeInForce.GTC

    def setup(self, config: Dict[str, Any]) -> None:
        self.hedge_threshold = float(config.get("hedge_threshold", self.hedge_threshold))

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        delta = float(features["portfolio_delta"])
        orders: List[Order] = []
        
        if delta >= self.hedge_threshold:
            orders.append(self.market_order(side=Side.SELL, qty=abs(delta), ctx=ctx, tif=self.tif))
        elif delta <= -self.hedge_threshold:
            orders.append(self.market_order(side=Side.BUY, qty=abs(delta), ctx=ctx, tif=self.tif))
            
        return orders
