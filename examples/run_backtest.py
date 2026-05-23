"""
Backtester completo listo para correr.

Única línea que cambia entre estrategias:
    from strategy.rmt_strategy import RMTStrategy   ← tu estrategia

Requisitos del dataset:
    Columnas obligatorias: timestamp | open | close
    Ordenado cronológicamente (o no — se ordena automáticamente).
"""
from __future__ import annotations

from pathlib import Path
from queue import Queue

import polars as pl

from engine.data_handler import DataFrameDataHandler
from engine.execution_handler import SimulatedExecutionHandler
from engine.event_loop import EventLoop
from engine.portfolio import SimplePortfolio
from pipeline.config import ValidationConfig
from pipeline.cpcv import CPCVConfig, CPCVEngine

# ── ÚNICO IMPORT QUE CAMBIA ───────────────────────────────────────────────
from strategy.rmt_strategy import RMTStrategy   # noqa: E402


# ── DATASET ──────────────────────────────────────────────────────────────
DATA_PATH = Path("data/dataset.parquet")   # parquet o csv
SYMBOL    = "RMT"


# ── PARÁMETROS DE LA ESTRATEGIA ───────────────────────────────────────────
# Se pasan directamente al constructor de RMTStrategy.
STRATEGY_PARAMS: dict = {
    # "ventana": 20,
    # "z_threshold": 1.5,
}


# ── EJECUCIÓN ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000.0
POSITION_PCT    = 0.02        # fracción del equity por trade (FixedFractionSizer)

EXECUTION = dict(
    slippage_pct        = 0.001,
    derecho_mercado_pct = 0.0006,
    arancel_alyc_pct    = 0.0003,
)


# ── VALIDACIÓN CPCV ──────────────────────────────────────────────────────
VAL_CFG = ValidationConfig(
    bars_per_year   = 252,     # 252 diario | 52 semanal | 12 mensual
    label_horizon   = 5,       # barras de purging (horizonte del label)
    embargo_pct     = 0.01,    # 1% del grupo post-test embargado
    half_life_days  = 365,     # decay para consenso de hiperparámetros
    n_trials        = 1,       # >1 activa DSR (penaliza p-hacking)
    block_bootstrap_reps = 0,  # 10_000 para bootstrap (lento pero robusto)
)

CPCV_CFG = CPCVConfig(
    n_groups      = 6,   # N: divide la serie en N bloques cronológicos
    n_test_groups = 2,   # k: C(6,2)=15 backtests, φ=5 trayectorias OOS
)


# ── RUNNER FACTORY ────────────────────────────────────────────────────────
def make_runner(params: dict):
    """
    Devuelve un BacktestRunner: callable(is_segments, oos_data) → (returns, signals).
    Se llama C(N,k)*k veces — cada vez reinicializa todo desde cero.
    """
    def runner(
        is_segments: list[pl.DataFrame],
        oos_data: pl.DataFrame,
    ) -> tuple[pl.Series, pl.Series]:

        queue    = Queue()
        strategy = RMTStrategy(queue, **params)

        # Fit sobre IS si la estrategia implementa fit()
        # (calibrar parámetros, ajustar modelos, etc.)
        if hasattr(strategy, "fit"):
            strategy.fit(is_segments)

        handler   = DataFrameDataHandler(queue, SYMBOL, oos_data)
        portfolio = SimplePortfolio(queue, INITIAL_CAPITAL, position_pct=POSITION_PCT)
        execution = SimulatedExecutionHandler(queue, **EXECUTION)
        loop      = EventLoop(queue, handler, strategy, portfolio, execution)
        loop.run()

        returns = portfolio.returns_series
        signals = pl.Series("signals", [1.0] * len(returns))  # reemplazar con señales reales si las tenés
        return returns, signals

    return runner


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
    print()

    engine = CPCVEngine(VAL_CFG, CPCV_CFG)
    report = engine.run(data, runner=make_runner(STRATEGY_PARAMS))

    print(report.summary())


if __name__ == "__main__":
    main()
