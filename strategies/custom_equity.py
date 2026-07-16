# custom strategy
from strategies.base import BaseSignalPolicy, SignalPosition
from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from typing import Any, Dict, List, Mapping

class MockTestStrategy(BaseSignalPolicy):
    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        return []
