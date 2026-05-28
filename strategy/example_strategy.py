from __future__ import annotations

import polars as pl

from strategy.base import Strategy


class MeanReversionStrategy(Strategy):
    """
    Z-score mean reversion sobre múltiples activos.

    Hiperparámetro (tuneado por CPCV): z_threshold
    Parámetro fijo:                    lookback

    fit() aprende media y std de precios del IS. En OOS, el z-score se calcula
    contra esas estadísticas IS (nivel de equilibrio aprendido), con fallback
    a ventana rolling si no hay fit previo.

    Señal por activo: long si z-score < -z_threshold (precio anormalmente bajo).
    Sizing: equal weight entre los activos con señal.
    """

    def __init__(self, z_threshold: float = 1.0, lookback: int = 20) -> None:
        self.z_threshold = z_threshold
        self.lookback = lookback
        self._fitted_stats: dict[str, tuple[float, float]] = {}

    def fit(self, is_segments: list[pl.DataFrame]) -> "MeanReversionStrategy":
        full_is = pl.concat(is_segments).sort("timestamp")
        symbols = [c for c in full_is.columns if c != "timestamp"]
        self._fitted_stats = {}
        for sym in symbols:
            prices = full_is[sym].drop_nulls()
            if len(prices) < 2:
                continue
            std = float(prices.std())
            if std > 0:
                self._fitted_stats[sym] = (float(prices.mean()), std)
        return self

    def get_weights(self, data: pl.DataFrame, positions: dict[str, float]) -> dict[str, float]:
        symbols = [c for c in data.columns if c != "timestamp"]

        if len(data) < self.lookback + 1:
            return {sym: 0.0 for sym in symbols}

        z_scores: dict[str, float] = {}
        for sym in symbols:
            prices = data[sym]
            current = float(prices[-1])

            if sym in self._fitted_stats:
                mean, std = self._fitted_stats[sym]
            else:
                window = prices[-self.lookback:]
                std = float(window.std())
                if not std:
                    continue
                mean = float(window.mean())

            z_scores[sym] = (current - mean) / std

        longs = []
        for sym in symbols:
            z   = z_scores.get(sym)
            pos = positions.get(sym, 0)

            if pos == 1:
                if z is not None and z < 0:
                    longs.append(sym)
            elif pos <= 0:
                if z is not None and z < -self.z_threshold:
                    longs.append(sym)

        w = 1.0 / len(longs) if longs else 0.0
        return {sym: (w if sym in longs else 0.0) for sym in symbols}
