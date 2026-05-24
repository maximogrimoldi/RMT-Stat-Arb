"""
Backtester completo listo para correr.

Única línea que cambia entre estrategias:
    from strategy.rmt_strategy import RMTStrategy   ← tu estrategia

Todo el proceso — tuning de hiperparámetros, fit sobre IS, backtest sobre OOS —
usa el engine completo: slippage, comisiones y fills realistas.

Requisitos del DataFrame:
    Columna 'timestamp' con las fechas.
    Una columna por símbolo con los precios de cierre.
    Ordenado cronológicamente.
"""
from __future__ import annotations

import polars as pl

from pipeline.config import ValidationConfig
from pipeline.cpcv import CPCVConfig, CPCVEngine
from pipeline.tuning import build_nested_cpcv_runner
from strategy.estimator import EventDrivenEstimator

# ── ÚNICO IMPORT QUE CAMBIA ───────────────────────────────────────────────
from strategy.example_strategy import MACrossStrategy


# ── HIPERPARÁMETROS (grilla para nested CPCV) ─────────────────────────────
PARAM_GRID = [
    {"fast_window": 5},
    {"fast_window": 10},
    {"fast_window": 20},
]

# ── PARÁMETROS FIJOS DE LA ESTRATEGIA ────────────────────────────────────
STRATEGY_PARAMS: dict = {
    "slow_window": 50,
}

# ── EJECUCIÓN ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL      = 100_000.0
REBALANCE_FREQUENCY  = "monthly"   # "daily" | "weekly" | "monthly"

EXECUTION = dict(
    slippage_pct        = 0.001,
    derecho_mercado_pct = 0.0006,
    arancel_alyc_pct    = 0.0003,
)

# ── VALIDACIÓN CPCV ───────────────────────────────────────────────────────
VAL_CFG = ValidationConfig(
    bars_per_year        = 252,
    label_horizon        = 5,
    embargo_pct          = 0.01,
    half_life_days       = 365,
    n_trials             = len(PARAM_GRID),
    block_bootstrap_reps = 0,
)

CPCV_CFG = CPCVConfig(
    n_groups      = 6,
    n_test_groups = 2,
)


# ── ESTIMATOR FACTORY ─────────────────────────────────────────────────────
def estimator_factory(params: dict) -> EventDrivenEstimator:
    return EventDrivenEstimator(
        strategy_factory    = MACrossStrategy,
        params              = {**STRATEGY_PARAMS, **params},
        initial_capital     = INITIAL_CAPITAL,
        rebalance_frequency = REBALANCE_FREQUENCY,
        **EXECUTION,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────
def main(data: pl.DataFrame) -> None:
    """
    data: DataFrame con columna 'timestamp' y una columna por símbolo con closes.
    """
    data = data.sort("timestamp")

    runner = build_nested_cpcv_runner(
        val_cfg           = VAL_CFG,
        grid              = PARAM_GRID,
        estimator_factory = estimator_factory,
        n_inner_splits    = 4,
    )

    engine = CPCVEngine(VAL_CFG, CPCV_CFG)
    report = engine.run(data, runner=runner)

    return report.metrics, report.equity_curves
