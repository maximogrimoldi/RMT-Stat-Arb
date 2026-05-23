from __future__ import annotations
from abc import ABC, abstractmethod

from engine.events import SignalEvent


class PositionSizer(ABC):
    @abstractmethod
    def size(
        self,
        signal: SignalEvent,
        equity: float,
        price: float,
        positions: dict[str, float],
    ) -> float:
        """Retorna la cantidad de unidades a operar para una entrada LONG o SHORT."""
        pass


class FixedFractionSizer(PositionSizer):
    """Apuesta una fracción fija del equity por trade."""

    def __init__(self, fraction: float = 0.02) -> None:
        self._fraction = fraction

    def size(
        self,
        signal: SignalEvent,
        equity: float,
        price: float,
        positions: dict[str, float],
    ) -> float:
        return (equity * self._fraction) / price
