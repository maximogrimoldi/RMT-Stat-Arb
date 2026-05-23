from __future__ import annotations
from abc import ABC, abstractmethod

import polars as pl


class Strategy(ABC):
    """
    Interfaz abstracta para todas las estrategias.
    Recibe historia de precios, devuelve weights. No sabe en qué split está.
    """

    @abstractmethod
    def get_weights(self, data: pl.DataFrame) -> dict[str, float]:
        """
        data: DataFrame [timestamp, close] con todas las barras hasta la actual.
        Devuelve {symbol: weight} donde weight es fracción del equity (negativo = short).
        Siempre incluir todos los símbolos operados; usar 0.0 para cerrar posición.
        """
        pass

