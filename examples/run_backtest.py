"""
Backtester completo listo para correr.

Única línea que cambia entre estrategias:
    from strategy.rmt_strategy import RMTStrategy   ← tu estrategia

Todo el proceso — tuning de hiperparámetros, fit sobre IS, backtest sobre OOS —
usa el engine completo: slippage, comisiones y fills al open reales.

Requisitos del dataset:
    Columnas obligatorias: timestamp | open | close
    Ordenado cronológicamente (o no — se ordena automáticamente).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from pipeline.config import ValidationConfig
from pipeline.cpcv import CPCVConfig, CPCVEngine
from pipeline.tuning import build_nested_cpcv_runner
from strategy.estimator import EventDrivenEstimator

# ── ÚNICO IMPORT QUE CAMBIA ───────────────────────────────────────────────
from strategy.rmt_strategy import RMTStrategy


# ── DATASET ───────────────────────────────────────────────────────────────
DATA_PATH = Path("data/dataset.parquet")   # parquet o csv
SYMBOL    = "RMT"


# ── HIPERPARÁMETROS (grilla para nested CPCV) ─────────────────────────────
# El tuning interno busca el mejor valor en cada fold IS.
# Si no tenés hiperparámetros que buscar, dejá un solo dict vacío.
PARAM_GRID = [
    # {"z_threshold": 1.0},
    # {"z_threshold": 1.5},
    # {"z_threshold": 2.0},
    {},   # sin hiperparámetros — el tuning corre igual pero con un solo candidato
]


# ── PARÁMETROS FIJOS DE LA ESTRATEGIA ────────────────────────────────────
# Se pasan siempre al constructor de RMTStrategy, independientemente del tuning.
STRATEGY_PARAMS: dict = {
    # "ventana": 20,
}


# ── EJECUCIÓN ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
POSITION_PCT    = 0.02   # fracción del equity por trade (FixedFractionSizer)

EXECUTION = dict(
    slippage_pct        = 0.001,
    derecho_mercado_pct = 0.0006,
    arancel_alyc_pct    = 0.0003,
)


# ── VALIDACIÓN CPCV ───────────────────────────────────────────────────────
VAL_CFG = ValidationConfig(
    bars_per_year        = 252,    # 252 diario | 52 semanal | 12 mensual
    label_horizon        = 5,      # barras de purging (horizonte del label)
    embargo_pct          = 0.01,   # 1% del grupo post-test embargado
    half_life_days       = 365,    # decay para consenso de hiperparámetros
    n_trials             = len(PARAM_GRID),   # activa DSR si hay múltiples candidatos
    block_bootstrap_reps = 0,      # 10_000 para bootstrap (lento pero robusto)
)

CPCV_CFG = CPCVConfig(
    n_groups      = 6,   # N: divide la serie en N bloques cronológicos
    n_test_groups = 2,   # k: C(6,2)=15 backtests, φ=5 trayectorias OOS
)


# ── ESTIMATOR FACTORY ─────────────────────────────────────────────────────
def estimator_factory(params: dict) -> EventDrivenEstimator:
    """
    Crea un estimador para cada combinación de hiperparámetros.
    params viene de PARAM_GRID — hiperparámetros que el tuning está evaluando.
    STRATEGY_PARAMS son los parámetros fijos que siempre se pasan.
    """
    return EventDrivenEstimator(
        strategy_factory = RMTStrategy,
        params           = {**STRATEGY_PARAMS, **params},
        symbol           = SYMBOL,
        initial_capital  = INITIAL_CAPITAL,
        position_pct     = POSITION_PCT,
        **EXECUTION,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {DATA_PATH}")

    if DATA_PATH.suffix.lower() in {".parquet", ".pq"}:
        data = pl.read_parquet(DATA_PATH)
    else:
        data = pl.read_csv(DATA_PATH, try_parse_dates=True)

    data = data.sort("timestamp")

    print(f"Dataset: {len(data)} barras  |  {data['timestamp'].min()} → {data['timestamp'].max()}")
    print(f"CPCV: N={CPCV_CFG.n_groups}, k={CPCV_CFG.n_test_groups}  →  "
          f"C({CPCV_CFG.n_groups},{CPCV_CFG.n_test_groups}) backtests, "
          f"φ={CPCV_CFG.n_groups - CPCV_CFG.n_test_groups} trayectorias")
    print(f"Grilla: {len(PARAM_GRID)} candidato(s)")
    print()

    runner = build_nested_cpcv_runner(
        val_cfg           = VAL_CFG,
        grid              = PARAM_GRID,
        estimator_factory = estimator_factory,
        n_inner_splits    = 4,
    )

    engine = CPCVEngine(VAL_CFG, CPCV_CFG)
    report = engine.run(data, runner=runner)

    print(report.summary())


if __name__ == "__main__":
    main()
