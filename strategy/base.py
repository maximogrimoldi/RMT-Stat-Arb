from __future__ import annotations
from abc import ABC, abstractmethod

import polars as pl


class Strategy(ABC):
    """
    Interfaz abstracta para todas las estrategias.
    Recibe historia de precios, devuelve weights. No sabe en qué split está.
    """

    @abstractmethod
    def get_weights(self, data: pl.DataFrame, positions: dict[str, float]) -> dict[str, float]:
        """
        data:      DataFrame [timestamp, close] con todas las barras hasta la actual.
        positions: unidades abiertas actualmente, ej. {"AAPL": 10.0, "MSFT": -5.0}.
        Devuelve {symbol: weight} donde weight es fracción del equity (negativo = short).
        Siempre incluir todos los símbolos operados; usar 0.0 para cerrar posición.
        """
        pass

