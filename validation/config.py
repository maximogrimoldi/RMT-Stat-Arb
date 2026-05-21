from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ValidationConfig:
    """
    Parámetros estadísticos compartidos por CPCVEngine.
    """

    # Bootstrap por bloques (para distribución del Sharpe)
    block_bootstrap_reps: int = 0
    alpha_halflife_bars: int | None = None   # longitud del bloque; None → se usa 20

    # Purging & Embargo (AFML Cap. 7)
    label_horizon: int = 1             # horizonte de la etiqueta en barras (h)
    embargo_pct: float = 0.01          # fracción del fold a embargar (fallback)
    embargo_bars: int | None = None    # barras absolutas de embargo; overridea embargo_pct si se setea

    # Frecuencia de los datos
    bars_per_year: int = 252    # 252 = diario, 52 = semanal, 12 = mensual

    # Deflated Sharpe Ratio
    n_trials: int = 1           # combinaciones de parámetros intentadas (para DSR)
