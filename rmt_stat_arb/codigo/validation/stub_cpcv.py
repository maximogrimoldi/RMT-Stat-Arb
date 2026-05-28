"""
STUB TEMPORAL de run_cpcv.

Devuelve datos sintéticos con la misma firma y estructura que devolverá
la implementación real del colaborador. Sirve para probar el flujo completo
de run_validation.py sin necesitar el backtester.

Para conectar la implementación real, reemplazá el import en run_validation.py:
    from validation.stub_cpcv import run_cpcv   # ← esto
    from validation.cpcv      import run_cpcv   # ← por esto
"""

import numpy as np
import pandas as pd


def run_cpcv(
    StrategyClass,
    data,
    grid,
    n_groups: int   = 10,
    n_test: int     = 2,
    commissions: float = 0.0001,
) -> dict:
    """
    STUB TEMPORAL de run_cpcv. Reemplazar por la implementación real
    del colaborador cuando esté lista.

    Parámetros
    ----------
    StrategyClass : clase de estrategia con interfaz get_weights(prices) y reset().
    data          : DataFrame de precios (fechas × activos) o None.
    grid          : lista de dicts de hiperparámetros (producida por param_grid()).
    n_groups      : número de grupos CPCV (default 10).
    n_test        : grupos usados como OOS en cada split (default 2).
    commissions   : comisión por operación (default 0.0001 = 1 bp).

    Devuelve
    --------
    dict con dos claves:
        "metrics"      : dict con sharpe_mean, sharpe_std, psr, dsr, n_paths,
                         consensus_params.
        "equity_curves": DataFrame con una columna por path (Path_1 … Path_n).
    """
    # 5 paths fijo mientras usamos el stub — suficiente para ver el gráfico
    # limpio sin saturar de líneas. El backtester real calculará el número
    # correcto: C(n_groups, n_test) * n_test / n_groups.
    n_paths = 5
    n_days  = len(data) if data is not None else 500

    rng = np.random.default_rng(42)

    equity: dict = {}
    sharpes: list = []
    for p in range(1, n_paths + 1):
        rets               = rng.normal(0.0004, 0.01, n_days)
        equity[f"Path_{p}"] = 100_000 * np.cumprod(1 + rets)
        sharpe             = float(np.mean(rets) / np.std(rets) * np.sqrt(252))
        sharpes.append(sharpe)

    equity_curves = pd.DataFrame(equity)

    sharpe_mean = float(np.mean(sharpes))
    sharpe_std  = float(np.std(sharpes))

    # PSR/DSR simplificados (stub)
    # PSR ≈ P(Sharpe_OOS > 0 | Sharpe_IS, n_obs, skew, kurt)
    # DSR ≈ PSR descontando el sesgo de selección por múltiples trials
    psr = float(np.clip(0.5 + sharpe_mean / 4, 0, 1))
    dsr = float(np.clip(psr - 0.03 * len(grid) / 100, 0, 1))

    # Parámetro modal de la grilla (el del medio como proxy del consenso)
    consensus_params = grid[len(grid) // 2] if grid else {}

    return {
        "metrics": {
            "sharpe_mean":      round(sharpe_mean, 4),
            "sharpe_std":       round(sharpe_std, 4),
            "psr":              round(psr, 4),
            "dsr":              round(dsr, 4),
            "n_paths":          n_paths,
            "consensus_params": consensus_params,
        },
        "equity_curves": equity_curves,
    }
