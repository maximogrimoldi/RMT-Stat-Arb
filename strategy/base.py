from __future__ import annotations
from abc import ABC, abstractmethod
from queue import Queue

from engine.events import MarketEvent


class Strategy(ABC):
    """
    Interfaz abstracta para todas las estrategias.
    Recibe datos, devuelve señales. No sabe en qué split está.
    El motor no implementa estrategias: las orquesta.
    """

    def __init__(self, events_queue: Queue) -> None:
        self._events_queue = events_queue

    @abstractmethod
    def on_market(self, event: MarketEvent) -> None:
        """Procesa un MarketEvent y puede emitir SignalEvents."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resetea el estado interno entre splits."""
        pass
