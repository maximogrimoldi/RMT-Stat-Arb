"""
RMTStrategy: adaptador de la estrategia RMT stat-arb para el backtester CPCV.

STATELESS: get_weights recalcula todo desde los precios recibidos.
Mismos inputs → mismos outputs, siempre. Crítico para CPCV.

CONTRATO DE NO LOOK-AHEAD:
  El backtester debe pasar data.iloc[:j] en el paso j, nunca más.
  Ver CONTRATO.md para detalles.
  Guarda opcional: get_weights(..., current_bar_date=data.index[j-1])
"""

import numpy as np
import pandas as pd

from strategy.signals import RMTStrategy as RMTPipeline


class RMTStrategy:
    """Estrategia RMT stat-arb. Stateless: mismos inputs → mismos outputs."""

    def __init__(
        self,
        entry_threshold:  float = 2.0,
        exit_threshold:   float = 0.5,
        ventana_betas:    int   = 252,
        ventana_zscore:   int   = 252,
        sizing_by_zscore: bool  = False,
    ):
        self.entry_threshold  = entry_threshold
        self.exit_threshold   = exit_threshold
        self.ventana_betas    = ventana_betas
        self.ventana_zscore   = ventana_zscore
        self.sizing_by_zscore = sizing_by_zscore
        self.rmt = RMTPipeline()

    def param_grid(self) -> list[dict]:
        """Grilla de hiperparámetros para el CPCV."""
        return [
            {"entry_threshold": e, "exit_threshold": x,
             "ventana_betas": vb, "ventana_zscore": vz,
             "sizing_by_zscore": s}
            for e  in [1.5, 2.0, 2.5]
            for x  in [1.0]
            for vb in [252]
            for vz in [252]
            for s  in [True, False]
        ]

    def _calcular_pesos(self, posiciones: dict, zs: pd.Series) -> dict:
        """
        Convierte {ticker: signo (+1/-1)} a {ticker: peso con signo}.

        Equiponderado (sizing_by_zscore=False): peso_i = signo / N
        Por z-score   (sizing_by_zscore=True):  peso_i = signo * |z_i| / Σ|z_j|
        En ambos casos Σ|peso_i| = 1.
        """
        if not posiciones:
            return {}

        if self.sizing_by_zscore:
            abs_z = {t: abs(float(zs[t])) for t in posiciones}
            total = sum(abs_z.values())
            if total == 0:                          # fallback si todos z=0
                n = len(posiciones)
                return {t: posiciones[t] / n for t in posiciones}
            return {t: posiciones[t] * abs_z[t] / total for t in posiciones}
        else:
            n = len(posiciones)
            return {t: posiciones[t] / n for t in posiciones}

    def get_weights(
        self,
        prices:             pd.DataFrame,
        current_positions:  "dict | None"           = None,
        current_bar_date:   "pd.Timestamp | None"   = None,
        return_diagnostics: bool                    = False,
    ) -> "dict | tuple[dict, dict]":
        """
        Recibe precios hasta la barra actual y posiciones abiertas.
        Devuelve {ticker: peso_objetivo} con signo y sizing.

        current_positions: {ticker: +1/-1}.
          +1 = long abierto, -1 = short abierto, ausente = sin posición.
          None equivale a dict vacío (primera barra, sin posiciones previas).
        current_bar_date: verifica prices.index[-1] == fecha; lanza ValueError
          si no coincide. Útil para detectar look-ahead durante desarrollo.
        return_diagnostics: si True, devuelve (weights, {"zscores": {ticker: float}})
          en lugar de solo weights. El path por defecto (False) es idéntico al anterior.

        Devuelve todo 0.0 durante calentamiento (< ventana_betas barras).
        """
        if current_positions is None:
            current_positions = {}

        # ── Guarda de contrato (opt-in) ───────────────────────────────────────
        if current_bar_date is not None:
            last = prices.index[-1]
            if last != pd.Timestamp(current_bar_date):
                raise ValueError(
                    f"[get_weights] Violación de contrato: "
                    f"prices.index[-1]={last.date()} "
                    f"!= current_bar_date={pd.Timestamp(current_bar_date).date()}. "
                    f"El backtester debe pasar data.iloc[:j] en el paso j."
                )

        tickers = list(prices.columns)
        vacío   = {t: 0.0 for t in tickers}

        # ── 1. Retornos ───────────────────────────────────────────────────────
        retornos = prices.pct_change().dropna()
        if len(retornos) < self.ventana_betas + 1:
            return (vacío, {"zscores": {}}) if return_diagnostics else vacío

        # ── 2. Residuos rolling (stateless) ──────────────────────────────────
        # calcular_residuos_rolling devuelve NaN en las primeras ventana_betas
        # filas; el resto son los residuos diarios sin lookahead.
        residuos_df, _, _ = self.rmt.calcular_residuos_rolling(
            retornos, ventana=self.ventana_betas
        )

        # ── 3. Z-score sobre la ventana de historial ──────────────────────────
        residuos_validos = residuos_df.dropna()
        if len(residuos_validos) < 2:
            return (vacío, {"zscores": {}}) if return_diagnostics else vacío

        ventana_z = residuos_validos.iloc[-self.ventana_zscore:]
        acum      = np.cumsum(ventana_z.values, axis=0)   # (ventana_z, N)
        zs        = pd.Series(self.rmt.zscore(acum), index=tickers)

        # ── 4. Lógica entry / exit ────────────────────────────────────────────
        posiciones_finales: dict = {}
        for t in tickers:
            z          = float(zs[t]) if not np.isnan(zs[t]) else 0.0
            pos_actual = current_positions.get(t, 0)

            if pos_actual != 0:
                # Cerrar si se cumple cualquiera de estas dos condiciones:
                #   (a) |z| < exit_threshold  → revirtió a la media
                #   (b) pos * z > 0           → z cruzó al lado opuesto a la apertura
                #       (abrí LONG porque z<0; si ahora z>0 se pasó de largo → cerrar)
                #       (abrí SHORT porque z>0; si ahora z<0 se pasó de largo → cerrar)
                # NO se da vuelta la posición: solo se cierra, no se abre la opuesta.
                z_cruzó = (pos_actual * z > 0)
                if abs(z) >= self.exit_threshold and not z_cruzó:
                    posiciones_finales[t] = pos_actual   # mantener
                # else → no incluir → se cierra
            else:
                # Sin posición: abrir si hay señal suficiente
                if z < -self.entry_threshold:
                    posiciones_finales[t] = 1    # long
                elif z > self.entry_threshold:
                    posiciones_finales[t] = -1   # short

        # ── 5. Sizing ─────────────────────────────────────────────────────────
        pesos = self._calcular_pesos(posiciones_finales, zs)

        # Devolver todos los tickers; los inactivos con peso 0.0
        weights = {t: pesos.get(t, 0.0) for t in tickers}
        if return_diagnostics:
            zscores = {t: (float(zs[t]) if not np.isnan(zs[t]) else None) for t in tickers}
            return weights, {"zscores": zscores}
        return weights

    def reset(self) -> None:
        """No-op: la estrategia es stateless, no hay estado que limpiar."""
        pass

    def __repr__(self) -> str:
        return (
            f"RMTStrategy(entry={self.entry_threshold}, exit={self.exit_threshold}, "
            f"ventana_betas={self.ventana_betas}, ventana_zscore={self.ventana_zscore}, "
            f"sizing={('zscore' if self.sizing_by_zscore else 'equal')})"
        )
