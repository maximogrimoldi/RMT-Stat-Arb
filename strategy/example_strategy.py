from __future__ import annotations

import polars as pl

from strategy.base import Strategy


class MACrossStrategy(Strategy):
    """
    Moving average crossover sobre múltiples activos.

    Hiperparámetro (tuneado por CPCV): fast_window
    Parámetro fijo:                    slow_window

    Señal por activo: long si MA(fast) > MA(slow), flat si no.
    Sizing: equal weight entre los activos con señal positiva.
    """

    def __init__(self, fast_window: int = 10, slow_window: int = 50) -> None:
        self.fast_window = fast_window
        self.slow_window = slow_window

    def get_weights(self, data: pl.DataFrame) -> dict[str, float]:
        symbols = [c for c in data.columns if c != "timestamp"]

        if len(data) < self.slow_window:
            return {sym: 0.0 for sym in symbols}

        longs = []
        for sym in symbols:
            prices   = data[sym]
            fast_ma  = prices[-self.fast_window:].mean()
            slow_ma  = prices[-self.slow_window:].mean()
            if fast_ma > slow_ma:
                longs.append(sym)

        w = 1.0 / len(longs) if longs else 0.0
        return {sym: (w if sym in longs else 0.0) for sym in symbols}
