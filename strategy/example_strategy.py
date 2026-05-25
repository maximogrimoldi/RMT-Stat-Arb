from __future__ import annotations

import polars as pl

from strategy.base import Strategy


class MeanReversionStrategy(Strategy):
    """
    Z-score mean reversion sobre múltiples activos.

    Hiperparámetro (tuneado por CPCV): z_threshold
    Parámetro fijo:                    lookback

    Señal por activo: long si z-score < -z_threshold (precio anormalmente bajo).
    Sizing: equal weight entre los activos con señal.
    """

    def __init__(self, z_threshold: float = 1.0, lookback: int = 20) -> None:
        self.z_threshold = z_threshold
        self.lookback = lookback

    def get_weights(self, data: pl.DataFrame, positions: dict[str, float]) -> dict[str, float]:
        symbols = [c for c in data.columns if c != "timestamp"]

        if len(data) < self.lookback + 1:
            return {sym: 0.0 for sym in symbols}

        z_scores: dict[str, float] = {}
        for sym in symbols:
            prices = data[sym]
            window = prices[-self.lookback:]
            std    = window.std()
            if not std:
                continue
            z_scores[sym] = (prices[-1] - window.mean()) / std

        longs = []
        for sym in symbols:
            z   = z_scores.get(sym)
            pos = positions.get(sym, 0)

            if pos == 1:
                # ya long: mantener hasta que el precio vuelva a la media
                if z is not None and z < 0:
                    longs.append(sym)
            elif pos <= 0:
                # flat o short (esta strat no opera short): entrar long si precio muy bajo
                if z is not None and z < -self.z_threshold:
                    longs.append(sym)

        w = 1.0 / len(longs) if longs else 0.0
        return {sym: (w if sym in longs else 0.0) for sym in symbols}
