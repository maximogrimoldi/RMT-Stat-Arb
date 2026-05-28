"""
run_validation_rmt.py — Conecta RMTStrategy con el motor CPCV real.

Patrón: examples/run_backtest.py del Backtester (no se toca el motor ni core.py).

Único cambio de interfaz: el motor pasa pl.DataFrame a get_weights, pero
RMTStrategy espera pd.DataFrame con DatetimeIndex. Lo resuelve RMTStrategyPolars,
una subclase delgada que convierte el input y delega en super().get_weights().
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import polars as pl

# ── Rutas absolutas ───────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_RMT_ROOT    = _SCRIPTS_DIR.parent                 # → rmt_stat_arb/
_CODIGO_DIR  = _RMT_ROOT / "codigo"               # RMTStrategy, data.*
_BACKTESTER  = _RMT_ROOT.parent / "Backtester"    # motor CPCV

# ── Orden de importación — resuelve conflicto de namespace "strategy/" ────────
#
# Ambos repos tienen un paquete llamado "strategy/":
#   codigo/strategy/     → core.py, signals.py
#   Backtester/strategy/ → base.py, estimator.py
#
# Estrategia:
#   1. Poner codigo/ PRIMERO en sys.path → "strategy" = paquete RMT.
#   2. Añadir Backtester/ al FINAL de sys.path (pipeline.* no tienen conflicto).
#   3. Inyectar los submodulos Backtester que el motor necesita (strategy.base,
#      strategy.estimator) directamente en sys.modules via importlib antes de
#      que se importen sus dependencias (engine.*).
#   Resultado: pipeline.*, engine.* se resuelven en Backtester/;
#              strategy.core se resuelve en codigo/;
#              strategy.base y strategy.estimator son los de Backtester/.

# Paso 1 — RMT
if str(_CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(_CODIGO_DIR))

from strategy.core import RMTStrategy          # strategy → codigo/strategy/
from data.ingest   import load_prices
from data.universe import UNIVERSE

# Paso 2 — Backtester al final (pipeline.*, engine.* sin conflicto)
if str(_BACKTESTER) not in sys.path:
    sys.path.append(str(_BACKTESTER))

# Paso 3 — Inyectar submodulos Backtester bajo el namespace "strategy"
def _load_bt(module_name: str, rel_path: str) -> None:
    """Carga un archivo de Backtester y lo registra en sys.modules."""
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        module_name, _BACKTESTER / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

_load_bt("strategy.base",      "strategy/base.py")
_load_bt("strategy.estimator", "strategy/estimator.py")

# Paso 4 — Motor CPCV (sin conflicto de nombres)
from pipeline.config    import ValidationConfig
from pipeline.cpcv      import CPCVConfig, CPCVEngine
from pipeline.tuning    import build_nested_cpcv_runner
from strategy.estimator import EventDrivenEstimator   # ya en sys.modules


# ── Wrapper Polars → pandas ───────────────────────────────────────────────────

class RMTStrategyPolars(RMTStrategy):
    """
    Subclase delgada: convierte el pl.DataFrame que pasa el motor al pd.DataFrame
    con DatetimeIndex que espera RMTStrategy. No modifica ninguna lógica.

    El motor llama:  get_weights(history: pl.DataFrame, positions: {ticker: +1/-1/0})
    RMTStrategy usa: get_weights(prices:  pd.DataFrame, current_positions=...)
    """
    def get_weights(self, data: pl.DataFrame, positions: dict) -> dict:
        prices_pd = data.to_pandas().set_index("timestamp")
        prices_pd.index = pd.to_datetime(prices_pd.index)
        return super().get_weights(prices_pd, current_positions=positions)


# ── Grid de hiperparámetros ───────────────────────────────────────────────────
# ventana_betas=60 / ventana_zscore=60 reduce el calentamiento de 252 a 60 días
PARAM_GRID: list[dict] = [
    {"entry_threshold": e, "exit_threshold": 1.0,
     "ventana_betas": 60, "ventana_zscore": 60,
     "sizing_by_zscore": s}
    for e in [1.5, 2.0, 2.5]
    for s in [True, False]
]


# ── Configuración ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL     = 100_000.0
REBALANCE_FREQUENCY = "monthly"     # stat-arb rebalanceo mensual

EXECUTION = dict(
    slippage_pct        = 0.001,
    derecho_mercado_pct = 0.0006,
    arancel_alyc_pct    = 0.0003,
)

VAL_CFG = ValidationConfig(
    bars_per_year        = 252,
    label_horizon        = 5,
    embargo_pct          = 0.01,
    n_trials             = len(PARAM_GRID),
    block_bootstrap_reps = 0,
    half_life_days       = 365.0,
)

CPCV_CFG = CPCVConfig(
    n_groups      = 6,
    n_test_groups = 2,
)

_RESULTS_DIR = _RMT_ROOT / "results" / "cpcv_v60"


# ── Estimator factory ─────────────────────────────────────────────────────────

def estimator_factory(params: dict) -> EventDrivenEstimator:
    """
    Cada llamada del tuner crea un EventDrivenEstimator fresco con los params
    de esa combinación. params viene directamente de PARAM_GRID — contiene
    todos los argumentos de RMTStrategy.__init__.
    """
    return EventDrivenEstimator(
        strategy_factory    = RMTStrategyPolars,
        params              = params,
        initial_capital     = INITIAL_CAPITAL,
        rebalance_frequency = REBALANCE_FREQUENCY,
        **EXECUTION,
    )


# ── Helpers de datos ──────────────────────────────────────────────────────────

def _load_prices_polars() -> pl.DataFrame:
    """Carga prices.parquet (pandas) y lo convierte al formato que espera CPCVEngine."""
    prices_pd = load_prices()[UNIVERSE]
    # reset_index mueve DatetimeIndex → columna; el nombre del índice es "Date" (yfinance)
    col_name  = prices_pd.index.name or "index"
    prices_pl = (
        pl.from_pandas(prices_pd.reset_index())
        .rename({col_name: "timestamp"})
        .with_columns(pl.col("timestamp").cast(pl.Date))
        .sort("timestamp")
    )
    return prices_pl


def _fetch_benchmark(start: str, end: str) -> pl.DataFrame | None:
    try:
        import yfinance as yf
        raw = yf.download("^GSPC", start=start, end=end,
                          auto_adjust=True, progress=False)["Close"]
        raw = raw.reset_index()
        raw.columns = ["timestamp", "close"]
        return (
            pl.from_pandas(raw)
            .with_columns(pl.col("timestamp").cast(pl.Date))
            .sort("timestamp")
            .with_columns(pl.col("close").pct_change().alias("mkt_return"))
            .drop_nulls("mkt_return")
            .select(["timestamp", "mkt_return"])
        )
    except Exception as e:
        print(f"[!] Sin benchmark S&P500 (backtest corre igual): {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Grid ─────────────────────────────────────────────────────────────────
    print(f"\n[*] Grid RMT: {len(PARAM_GRID)} combinaciones")
    for i, p in enumerate(PARAM_GRID, 1):
        print(f"    {i}. entry={p['entry_threshold']}  exit={p['exit_threshold']}"
              f"  sizing={'zscore' if p['sizing_by_zscore'] else 'equal'}")

    # ── Datos ─────────────────────────────────────────────────────────────────
    print("\n[*] Cargando precios desde disco…")
    data = _load_prices_polars()
    print(f"[*] {data.shape[1] - 1} tickers × {len(data)} días "
          f"({data['timestamp'].min()} → {data['timestamp'].max()})")

    start = str(data["timestamp"].min())[:10]
    end   = str(data["timestamp"].max())[:10]
    mkt   = _fetch_benchmark(start, end)
    if mkt is not None:
        print(f"[*] Benchmark: {len(mkt)} días")

    # ── CPCV ──────────────────────────────────────────────────────────────────
    print("\n[*] Construyendo runner nested CPCV…")
    runner = build_nested_cpcv_runner(
        val_cfg           = VAL_CFG,
        grid              = PARAM_GRID,
        estimator_factory = estimator_factory,
        n_inner_splits    = 4,
    )

    engine = CPCVEngine(VAL_CFG, CPCV_CFG)
    print("[*] Corriendo CPCV (puede tardar varios minutos)…")
    report = engine.run(data, runner=runner, market_data=mkt)

    # ── Métricas ──────────────────────────────────────────────────────────────
    m = report.metrics
    print("\n" + "═"*52)
    print("  RESULTADO CPCV — RMT Stat-Arb")
    print("═"*52)
    print(f"  N paths           : {m['phi']}")
    print(f"  Sharpe medio      : {m['sharpe_mean']:.3f}")
    print(f"  Sharpe std        : {m['sharpe_std']:.3f}")
    print(f"  DSR               : {m['dsr']:.3f}")
    print(f"  Max Drawdown      : {m['max_drawdown']:.2%}")
    print(f"  % paths positivos : {m['pct_positive_paths']:.1%}")
    if "market_regression" in m:
        mr = m["market_regression"]
        print(f"  Alpha (anual)     : {mr.get('alpha_annualized', float('nan')):.3f}")
        print(f"  Beta              : {mr.get('beta', float('nan')):.3f}")
    print("═"*52)

    # ── Guardar equity curves ─────────────────────────────────────────────────
    rows = []
    for i, ec in enumerate(report.equity_curves, 1):
        for row in ec.iter_rows(named=True):
            rows.append({"path": i, "bar": row["bar"], "equity": row["equity"]})
    ec_df = pd.DataFrame(rows)
    ec_path = _RESULTS_DIR / "equity_curves_cpcv.parquet"
    ec_df.to_parquet(ec_path)
    print(f"\n[*] Equity curves guardadas en {ec_path}")


if __name__ == "__main__":
    main()
