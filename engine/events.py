from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


EventType = Literal["MARKET", "SIGNAL", "ORDER", "FILL"]
SignalDirection = Literal["LONG", "SHORT", "EXIT"]
OrderDirection = Literal["LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"]
OrderType = Literal["MKT", "LMT"]


@dataclass
class MarketEvent:
    type: EventType = field(default="MARKET", init=False)
    timestamp: datetime
    symbol: str
    open: float
    close: float


@dataclass
class SignalEvent:
    type: EventType = field(default="SIGNAL", init=False)
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    strength: float  # [-1.0, 1.0]


@dataclass
class OrderEvent:
    type: EventType = field(default="ORDER", init=False)
    timestamp: datetime
    symbol: str
    order_type: OrderType
    quantity: float
    direction: OrderDirection


@dataclass
class FillEvent:
    type: EventType = field(default="FILL", init=False)
    timestamp: datetime
    symbol: str
    direction: OrderDirection
    quantity: float
    fill_price: float
    commission: float
    slippage: float
