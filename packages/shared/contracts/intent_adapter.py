# -*- coding: utf-8 -*-
"""
Intent Adapter - Migration from Legacy OrderIntent.

Provides adapter functions to convert between legacy OrderIntent
(from core_models.py) and new OrderIntent (from contracts/intent.py).

This enables gradual migration of existing strategies to the new
intent format without breaking changes.

Legacy Format (core_models.py):
    - ts: int (unix ms)
    - symbol: str
    - side: Side (BUY/SELL)
    - order_type: OrderType (MARKET/LIMIT)
    - volume_frac: Decimal [-1..1]
    - price_offset_ticks: int
    - time_in_force: TimeInForce
    - client_tag: str
    - meta: Dict

New Format (contracts/intent.py):
    - strategy_id: str
    - symbol: str
    - intent_type: IntentType (MARKET_ENTRY/LIMIT_EXIT/etc.)
    - side: IntentSide (LONG/SHORT/FLAT)
    - target_quantity: Optional[Decimal]
    - target_notional: Optional[Decimal]
    - limit_price: Optional[Decimal]
    - stop_price: Optional[Decimal]
    - time_in_force: str
    - urgency: IntentPriority
    - reason: str
    - metadata: Dict
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from packages.shared.contracts.intent import (
    OrderIntent,
    IntentType,
    IntentSide,
    IntentPriority,
)


@dataclass
class LegacyIntentContext:
    """
    Context required for legacy intent conversion.

    Legacy intents use volume_frac which requires context to convert
    to absolute quantities.
    """

    # Strategy info
    strategy_id: str = "legacy_strategy"

    # Position context for volume_frac calculation
    max_position_size: Decimal = Decimal("10000")
    current_position: Decimal = Decimal("0")

    # Price context for ticks calculation
    current_price: Decimal = Decimal("0")
    tick_size: Decimal = Decimal("0.01")


class IntentAdapter:
    """
    Adapter for converting between legacy and new OrderIntent formats.

    Usage:
        adapter = IntentAdapter()

        # Convert legacy to new
        context = LegacyIntentContext(
            strategy_id="my_strategy",
            max_position_size=Decimal("1000"),
            current_price=Decimal("50000"),
        )
        new_intent = adapter.from_legacy(legacy_intent, context)

        # Convert new to legacy (for backwards compatibility)
        legacy_intent = adapter.to_legacy(new_intent)
    """

    @staticmethod
    def from_legacy(
        legacy: Dict[str, Any],
        context: LegacyIntentContext,
    ) -> OrderIntent:
        """
        Convert legacy OrderIntent dict to new OrderIntent.

        Args:
            legacy: Legacy intent as dictionary
            context: Conversion context

        Returns:
            New format OrderIntent
        """
        # Extract legacy fields
        symbol = str(legacy.get("symbol", ""))
        legacy_side = str(legacy.get("side", "BUY"))
        legacy_order_type = str(legacy.get("order_type", "MARKET"))
        volume_frac = Decimal(str(legacy.get("volume_frac", "0")))
        price_offset_ticks = int(legacy.get("price_offset_ticks", 0))
        time_in_force = str(legacy.get("time_in_force", "GTC"))
        client_tag = str(legacy.get("client_tag", ""))
        meta = dict(legacy.get("meta", {}))

        # Convert side
        if legacy_side == "BUY":
            intent_side = IntentSide.LONG
        elif legacy_side == "SELL":
            intent_side = IntentSide.SHORT
        else:
            intent_side = IntentSide.FLAT

        # Determine intent type based on order_type and current position
        if volume_frac == 0:
            intent_type = IntentType.HOLD
        elif legacy_order_type == "MARKET":
            # Determine if entry or exit based on position change direction
            if volume_frac > 0 and context.current_position >= 0:
                intent_type = IntentType.MARKET_ENTRY
            elif volume_frac < 0 and context.current_position <= 0:
                intent_type = IntentType.MARKET_ENTRY
            else:
                intent_type = IntentType.MARKET_EXIT
        elif legacy_order_type == "LIMIT":
            if volume_frac > 0 and context.current_position >= 0:
                intent_type = IntentType.LIMIT_ENTRY
            elif volume_frac < 0 and context.current_position <= 0:
                intent_type = IntentType.LIMIT_ENTRY
            else:
                intent_type = IntentType.LIMIT_EXIT
        else:
            intent_type = IntentType.HOLD

        # Calculate target quantity from volume_frac
        target_quantity: Optional[Decimal] = None
        if volume_frac != 0:
            target_quantity = abs(volume_frac) * context.max_position_size

        # Calculate limit price from price_offset_ticks
        limit_price: Optional[Decimal] = None
        if price_offset_ticks != 0 and context.current_price > 0:
            limit_price = context.current_price + (
                Decimal(price_offset_ticks) * context.tick_size
            )

        # Create new intent
        return OrderIntent(
            strategy_id=context.strategy_id,
            symbol=symbol,
            intent_type=intent_type,
            side=intent_side,
            target_quantity=target_quantity,
            limit_price=limit_price,
            time_in_force=time_in_force,
            urgency=IntentPriority.NORMAL,
            reason=f"Migrated from legacy: {client_tag}",
            metadata={
                **meta,
                "legacy_client_tag": client_tag,
                "legacy_volume_frac": str(volume_frac),
                "legacy_price_offset_ticks": price_offset_ticks,
            },
        )

    @staticmethod
    def to_legacy(
        intent: OrderIntent,
        context: LegacyIntentContext,
    ) -> Dict[str, Any]:
        """
        Convert new OrderIntent to legacy format dict.

        Args:
            intent: New format OrderIntent
            context: Conversion context

        Returns:
            Legacy format as dictionary
        """
        import time

        # Convert side
        if intent.side == IntentSide.LONG:
            legacy_side = "BUY"
        elif intent.side == IntentSide.SHORT:
            legacy_side = "SELL"
        else:
            legacy_side = "SELL" if context.current_position > 0 else "BUY"

        # Convert order type
        if intent.is_limit:
            legacy_order_type = "LIMIT"
        else:
            legacy_order_type = "MARKET"

        # Calculate volume_frac from target_quantity
        volume_frac = Decimal("0")
        if intent.target_quantity and context.max_position_size > 0:
            volume_frac = intent.target_quantity / context.max_position_size
            if intent.side == IntentSide.SHORT:
                volume_frac = -volume_frac

        # Calculate price_offset_ticks from limit_price
        price_offset_ticks = 0
        if intent.limit_price and context.current_price > 0 and context.tick_size > 0:
            price_offset_ticks = int(
                (intent.limit_price - context.current_price) / context.tick_size
            )

        return {
            "ts": int(intent.created_at.timestamp() * 1000),
            "symbol": intent.symbol,
            "side": legacy_side,
            "order_type": legacy_order_type,
            "volume_frac": str(volume_frac),
            "price_offset_ticks": price_offset_ticks,
            "time_in_force": intent.time_in_force,
            "client_tag": str(intent.intent_id),
            "meta": {
                **intent.metadata,
                "original_intent_id": str(intent.intent_id),
                "original_intent_type": intent.intent_type.value,
            },
        }

    @staticmethod
    def is_legacy_format(data: Dict[str, Any]) -> bool:
        """
        Check if data is in legacy format.

        Args:
            data: Intent data as dictionary

        Returns:
            True if legacy format
        """
        legacy_fields = {"ts", "side", "order_type", "volume_frac"}
        new_fields = {"strategy_id", "intent_type", "target_quantity"}

        has_legacy = len(legacy_fields & set(data.keys())) >= 3
        has_new = len(new_fields & set(data.keys())) >= 2

        return has_legacy and not has_new

    @staticmethod
    def normalize(
        data: Dict[str, Any],
        context: LegacyIntentContext,
    ) -> OrderIntent:
        """
        Normalize any intent format to new OrderIntent.

        Args:
            data: Intent data (legacy or new format)
            context: Conversion context

        Returns:
            New format OrderIntent
        """
        if IntentAdapter.is_legacy_format(data):
            return IntentAdapter.from_legacy(data, context)
        else:
            return OrderIntent.from_dict(data)


class DecisionToIntentAdapter:
    """
    Adapter for converting Decision objects to OrderIntent.

    Some legacy code may use "Decision" pattern instead of OrderIntent.
    This adapter handles that migration.
    """

    @staticmethod
    def from_decision(
        decision: Dict[str, Any],
        strategy_id: str,
        symbol: str,
    ) -> OrderIntent:
        """
        Convert a decision dict to OrderIntent.

        Decisions are higher-level than intents and may contain
        action recommendations rather than specific order parameters.

        Args:
            decision: Decision dictionary
            strategy_id: Strategy ID
            symbol: Symbol for the decision

        Returns:
            OrderIntent
        """
        action = str(decision.get("action", "HOLD")).upper()
        confidence = Decimal(str(decision.get("confidence", "0.5")))
        target = decision.get("target_position", decision.get("target"))

        # Map action to intent type
        action_mapping = {
            "BUY": IntentType.MARKET_ENTRY,
            "SELL": IntentType.MARKET_EXIT,
            "LONG": IntentType.MARKET_ENTRY,
            "SHORT": IntentType.MARKET_ENTRY,
            "CLOSE": IntentType.CLOSE_POSITION,
            "FLATTEN": IntentType.FLATTEN_ALL,
            "HOLD": IntentType.HOLD,
            "NO_ACTION": IntentType.NO_ACTION,
        }

        intent_type = action_mapping.get(action, IntentType.HOLD)

        # Map action to side
        side_mapping = {
            "BUY": IntentSide.LONG,
            "LONG": IntentSide.LONG,
            "SELL": IntentSide.SHORT,
            "SHORT": IntentSide.SHORT,
            "CLOSE": IntentSide.FLAT,
            "FLATTEN": IntentSide.FLAT,
            "HOLD": IntentSide.FLAT,
        }

        side = side_mapping.get(action, IntentSide.FLAT)

        # Map confidence to urgency
        if confidence >= Decimal("0.9"):
            urgency = IntentPriority.URGENT
        elif confidence >= Decimal("0.75"):
            urgency = IntentPriority.HIGH
        elif confidence >= Decimal("0.5"):
            urgency = IntentPriority.NORMAL
        else:
            urgency = IntentPriority.LOW

        # Target quantity
        target_quantity = None
        if target is not None:
            target_quantity = abs(Decimal(str(target)))

        return OrderIntent(
            strategy_id=strategy_id,
            symbol=symbol,
            intent_type=intent_type,
            side=side,
            target_quantity=target_quantity,
            urgency=urgency,
            reason=decision.get("reason", f"Decision action: {action}"),
            metadata={
                "original_decision": decision,
                "confidence": str(confidence),
            },
        )


# Convenience functions for simple usage
def migrate_legacy_intent(
    legacy: Dict[str, Any],
    strategy_id: str = "legacy",
    max_position: Decimal = Decimal("10000"),
    current_price: Decimal = Decimal("0"),
) -> OrderIntent:
    """
    Simple function to migrate a legacy intent.

    Args:
        legacy: Legacy intent dictionary
        strategy_id: Strategy ID to use
        max_position: Maximum position for volume_frac calculation
        current_price: Current price for tick calculation

    Returns:
        New format OrderIntent
    """
    context = LegacyIntentContext(
        strategy_id=strategy_id,
        max_position_size=max_position,
        current_price=current_price,
    )
    return IntentAdapter.from_legacy(legacy, context)


def is_valid_new_intent(data: Dict[str, Any]) -> bool:
    """
    Check if data is valid new intent format.

    Args:
        data: Intent data

    Returns:
        True if valid new format
    """
    required_fields = {"strategy_id", "symbol", "intent_type", "side"}
    return required_fields <= set(data.keys())
