from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


OrderDirection = Literal["LONG", "SHORT", "EXIT_LONG", "EXIT_SHORT"]


@dataclass
class OrderEvent:
    timestamp: datetime
    symbol: str
    quantity: float
    direction: OrderDirection


@dataclass
class FillEvent:
    timestamp: datetime
    symbol: str
    direction: OrderDirection
    quantity: float
    fill_price: float
    commission: float
    slippage: float
