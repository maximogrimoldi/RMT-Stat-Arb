"""
run_validation_rmt.py — Conecta RMTStrategy con el motor CPCV.

Arquitectura precompute (López de Prado):
  1. calcular_residuos_rolling corre UNA VEZ sobre el dataset completo → _RESIDUOS_PD.
  2. En cada barra OOS, get_weights() slicéa _RESIDUOS_PD hasta la fecha actual.
  3. El engine sigue usando precios reales para contabilidad (fills, notional).

Único cambio de interfaz: el motor pasa pl.DataFrame; RMTStrategy espera
pd.DataFrame con DatetimeIndex. Lo resuelve RMTStrategyPolarsPrecomputed.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from math import comb as _comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

# ── Rutas absolutas ───────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_RMT_ROOT    = _SCRIPTS_DIR.parent          # → rmt_stat_arb/
_CPCV_DIR    = _RMT_ROOT.parent / "cpcv"   # → motor CPCV

# ── sys.path ──────────────────────────────────────────────────────────────────
# rmt_stat_arb/ primero → strategy.core, data.*, engines.*, monitoring.*
# cpcv/ al final       → pipeline.*, engine.*, analysis.*
# pipeline/, engine/, analysis/ ya no colisionan (solo existen en cpcv/).
# strategy/ sigue colisionando (ambos dirs se llaman "strategy/"), pero el
# inject es mínimo: solo strategy.base y strategy.estimator desde cpcv/.
if str(_RMT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RMT_ROOT))
if str(_CPCV_DIR) not in sys.path:
    sys.path.append(str(_CPCV_DIR))

from strategy.core import RMTStrategy
from data.ingest   import load_prices
from data.universe import UNIVERSE

def _inject(name: str, path: Path) -> None:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)

_inject("strategy.base",      _CPCV_DIR / "strategy/base.py")
_inject("strategy.estimator", _CPCV_DIR / "strategy/estimator.py")

from pipeline.config    import ValidationConfig
from pipeline.cpcv      import CPCVConfig, CPCVEngine
from pipeline.tuning    import build_nested_cpcv_runner, tune_inner_is_segments, _fit_estimator
from strategy.estimator import EventDrivenEstimator


# ── Helpers de conversión ─────────────────────────────────────────────────────

def _pl_to_pd(data: pl.DataFrame) -> pd.DataFrame:
    df = data.to_pandas().set_index("timestamp")
    df.index = pd.to_datetime(df.index)
    return df

# Residuos pre-computados compartidos entre todas las instancias de la corrida
_RESIDUOS_PD: "pd.DataFrame | None" = None

# Consensus params capturados por fold externo del CPCV
_consensus_log: list[dict] = []


class RMTStrategyPolarsPrecomputed(RMTStrategy):
    """
    Para CPCV con precompute.

    precompute(): corre calcular_residuos_rolling una vez sobre el dataset
      completo y guarda los residuos en _RESIDUOS_PD. Devuelve los PRECIOS
      sin cambiar — el engine los sigue usando para contabilidad (fills, notional).

    get_weights(): ignora la historia de precios recibida y busca los residuos
      hasta la fecha actual en _RESIDUOS_PD. Así el portfolio ve precios reales
      y la estrategia ve residuos pre-computados.
    """
    def precompute(self, data: pl.DataFrame) -> pl.DataFrame:
        global _RESIDUOS_PD
        residuos_pd = super().precompute(_pl_to_pd(data))
        _RESIDUOS_PD = residuos_pd
        return data  # precios sin cambiar

    def get_weights(self, data: pl.DataFrame, positions: dict) -> dict:
        tickers = [c for c in data.columns if c != "timestamp"]
        vacío   = {t: 0.0 for t in tickers}

        if _RESIDUOS_PD is None:
            return vacío

        current_date   = pd.Timestamp(data["timestamp"].tail(1)[0])
        residuos_slice = _RESIDUOS_PD[_RESIDUOS_PD.index <= current_date]
        return super()._get_weights_from_residuals(residuos_slice, positions)


# ── Grid de hiperparámetros ───────────────────────────────────────────────────
PARAM_GRID: list[dict] = [
    {"entry_threshold": e, "exit_threshold": 1.0,
     "ventana_betas": 252, "ventana_zscore": 252,
     "sizing_by_zscore": s}
    for e in [1.5, 2.0, 2.5]
    for s in [True, False]
]


# ── Configuración ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL     = 100_000.0
REBALANCE_FREQUENCY = "monthly"

EXECUTION = dict(
    slippage_pct        = 0.001,
    derecho_mercado_pct = 0.0006,
    arancel_alyc_pct    = 0.0003,
)

VAL_CFG = ValidationConfig(
    bars_per_year        = 252,
    label_horizon        = 5,
    embargo_bars         = 25,
    n_trials             = len(PARAM_GRID),
    block_bootstrap_reps = 0,
    half_life_days       = 365.0,
)

CPCV_CFG = CPCVConfig(
    n_groups      = 6,
    n_test_groups = 2,
)

_RESULTS_DIR = _RMT_ROOT / "results" / "backtesting"
_N_INNER_SPLITS = 4


# ── Estimator factory ─────────────────────────────────────────────────────────

def estimator_factory(params: dict) -> EventDrivenEstimator:
    """
    Usa RMTStrategyPolarsPrecomputed: get_weights() recibe residuos pre-computados
    y solo calcula z-score + entry/exit, sin recorrer calcular_residuos_rolling.
    """
    return EventDrivenEstimator(
        strategy_factory    = RMTStrategyPolarsPrecomputed,
        params              = params,
        initial_capital     = INITIAL_CAPITAL,
        rebalance_frequency = REBALANCE_FREQUENCY,
        **EXECUTION,
    )


# ── Consensus logging ────────────────────────────────────────────────────────

def _wrap_runner_with_consensus_log(original_runner):
    """
    Replica el closure del runner original capturando tuning.consensus_params
    antes del del tuning. Preserva el atributo .precompute si existe.
    """
    freevars = original_runner.__code__.co_freevars
    cells    = {name: cell.cell_contents
                for name, cell in zip(freevars, original_runner.__closure__)}

    _val_cfg           = cells["val_cfg"]
    _grid              = cells["grid"]
    _estimator_factory = cells["estimator_factory"]
    _n_inner_splits    = cells["n_inner_splits"]
    _score_fn          = cells["score_fn"]

    def wrapped_runner(is_segments, oos_data):
        tuning = tune_inner_is_segments(
            is_segments=is_segments,
            val_cfg=_val_cfg,
            grid=_grid,
            estimator_factory=_estimator_factory,
            n_splits=_n_inner_splits,
            score_fn=_score_fn,
        )
        params = tuning.consensus_params
        _consensus_log.append(dict(params))

        estimator = _estimator_factory(params)
        try:
            fitted   = _fit_estimator(estimator, is_segments)
            returns, signals = fitted.predict(oos_data)
            return returns, signals
        finally:
            del estimator
            del tuning

    wrapped_runner.precompute = getattr(original_runner, "precompute", None)
    return wrapped_runner


def aggregate_consensus(log: list[dict]) -> dict:
    """
    Agrega N consensus_params de los folds externos del CPCV.
    Continuos: mediana. Discretos (sizing_by_zscore): moda; empate → primer
    valor del PARAM_GRID (criterio determinístico).
    """
    import statistics as _st
    if not log:
        return {}

    _discrete = {"sizing_by_zscore"}
    result: dict = {}

    for key in log[0].keys():
        values = [fold[key] for fold in log if key in fold]
        if not values:
            continue
        if key in _discrete:
            counts: dict = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            max_count  = max(counts.values())
            candidates = {v for v, c in counts.items() if c == max_count}
            for p in PARAM_GRID:
                if p[key] in candidates:
                    result[key] = p[key]
                    break
        else:
            med = _st.median(values)
            result[key] = int(round(med)) if all(isinstance(v, int) for v in values) else med

    return result


# ── Helpers de datos ──────────────────────────────────────────────────────────

def _load_prices_polars() -> pl.DataFrame:
    """Carga prices.parquet (pandas) y lo convierte al formato que espera CPCVEngine."""
    prices_pd = load_prices()[UNIVERSE]
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


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_equity_curves(ec_df: pd.DataFrame, out_path: Path) -> None:
    """Genera PNG con las N equity curves del CPCV (una línea por path)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for path_id in sorted(ec_df["path"].unique()):
        eq = ec_df[ec_df["path"] == path_id]["equity"].values
        ax.plot(eq, label=f"Path {path_id}", linewidth=1.2)
    ax.set_title("CPCV Equity Curves — RMT Stat-Arb", fontsize=13, fontweight="bold")
    ax.set_xlabel("Barra")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    # ── 1. Grid ───────────────────────────────────────────────────────────────
    print(f"\n[*] Grid RMT: {len(PARAM_GRID)} combinaciones")
    for i, p in enumerate(PARAM_GRID, 1):
        print(f"    {i}. entry={p['entry_threshold']}  exit={p['exit_threshold']}"
              f"  sizing={'zscore' if p['sizing_by_zscore'] else 'equal'}")

    # ── 2. CPCVConfig + n_inner_splits ────────────────────────────────────────
    from pipeline.cpcv import n_paths as _n_paths
    phi = _n_paths(CPCV_CFG.n_groups, CPCV_CFG.n_test_groups)
    print(f"\n[*] CPCVConfig: n_groups={CPCV_CFG.n_groups}  n_test_groups={CPCV_CFG.n_test_groups}"
          f"  →  φ={phi} paths")
    print(f"[*] n_inner_splits={_N_INNER_SPLITS}  |  embargo_bars={VAL_CFG.embargo_bars}"
          f"  |  rebalance={REBALANCE_FREQUENCY}")

    # ── 3. Datos ──────────────────────────────────────────────────────────────
    print("\n[*] Cargando precios desde disco…")
    data = _load_prices_polars()
    print(f"[*] {data.shape[1] - 1} tickers × {len(data)} días "
          f"({data['timestamp'].min()} → {data['timestamp'].max()})")

    start = str(data["timestamp"].min())[:10]
    end   = str(data["timestamp"].max())[:10]
    mkt   = _fetch_benchmark(start, end)
    if mkt is not None:
        print(f"[*] Benchmark: {len(mkt)} días")

    # ── 3b. Bloque de configuración ───────────────────────────────────────────
    n_tickers = data.shape[1] - 1
    n_dias    = len(data)
    n_combos  = _comb(CPCV_CFG.n_groups, CPCV_CFG.n_test_groups)
    slip      = EXECUTION["slippage_pct"]
    dmkt      = EXECUTION["derecho_mercado_pct"]
    alyc      = EXECUTION["arancel_alyc_pct"]
    total_rt  = (slip + dmkt + alyc) * 2

    print("\n" + "═"*51)
    print("  CONFIGURACIÓN DEL BACKTEST")
    print("═"*51)
    print(f"  Universo de datos     : {n_tickers} tickers líquidos del S&P 500")
    print(f"  Período               : {start} → {end} ({n_dias} días)")
    print(f"  Fuente de datos       : Yahoo Finance (yfinance)")
    print(f"  Costos de transacción :")
    print(f"    - Slippage           : {slip*100:.2f}%")
    print(f"    - Derecho de mercado : {dmkt*100:.2f}%")
    print(f"    - Arancel ALYC       : {alyc*100:.2f}%")
    print(f"    - Total ida+vuelta   : ~{total_rt*100:.2f}%")
    print(f"  Rebalanceo            : {REBALANCE_FREQUENCY}")
    print(f"  Capital inicial       : ${INITIAL_CAPITAL:,.0f}")
    print(f"  Validación            : CPCV (Combinatorial Purged Cross-Validation)")
    print(f"    - N grupos          : {CPCV_CFG.n_groups}")
    print(f"    - Test groups       : {CPCV_CFG.n_test_groups}")
    print(f"    - Combinaciones     : {n_combos}")
    print(f"    - Trayectorias (φ)  : {phi}")
    print(f"    - Purge horizon     : {VAL_CFG.label_horizon} barras")
    print(f"    - Embargo           : {VAL_CFG.embargo_bars} barras")
    print(f"    - Nested splits     : {_N_INNER_SPLITS}")
    print("═"*51)

    # ── 4. Runner + wrapper de consensus ──────────────────────────────────────
    print("\n[*] Construyendo runner nested CPCV…")
    _precompute_strat = RMTStrategyPolarsPrecomputed(**PARAM_GRID[0])
    runner = build_nested_cpcv_runner(
        val_cfg           = VAL_CFG,
        grid              = PARAM_GRID,
        estimator_factory = estimator_factory,
        n_inner_splits    = _N_INNER_SPLITS,
        precompute_fn     = _precompute_strat.precompute,
    )
    runner = _wrap_runner_with_consensus_log(runner)

    # ── 5. Correr CPCV ────────────────────────────────────────────────────────
    engine = CPCVEngine(VAL_CFG, CPCV_CFG)
    print("[*] Corriendo CPCV (puede tardar varios minutos)…")
    report = engine.run(data, runner=runner, market_data=mkt)
    assert _RESIDUOS_PD is not None, "[ERROR] precompute no corrió — el hook no se enganchó"

    # ── 6. Agregación de consensus ─────────────────────────────────────────────
    best = aggregate_consensus(_consensus_log)

    # ── 7. Bloque RESULTADO CPCV ──────────────────────────────────────────────
    m  = report.metrics
    mr = m.get("market_regression", {})
    print("\n" + "═"*52)
    print("  RESULTADO CPCV — RMT Stat-Arb")
    print("═"*52)
    print(f"  N paths           : {m['phi']}")
    print(f"  Sharpe medio      : {m['sharpe_mean']:.3f}")
    print(f"  Sharpe std        : {m['sharpe_std']:.3f}")
    print(f"  DSR               : {m['dsr']:.3f}")
    print(f"  Max Drawdown      : {m['max_drawdown']:.2%}")
    print(f"  Turnover anual    : {m['turnover_annual_mean']:.2f}x")
    print(f"  % paths positivos : {m['pct_positive_paths']:.1%}")
    if mr:
        print(f"  Alpha (anual)     : {mr.get('alpha', float('nan')):.3f}")
        print(f"  Beta              : {mr.get('beta', float('nan')):.3f}")
    print("═"*52)

    # ── 8. Bloque CONSENSO POR FOLD ───────────────────────────────────────────
    print("\n" + "═"*52)
    print(f"  CONSENSO POR FOLD ({len(_consensus_log)} folds del CPCV externo)")
    print("═"*52)
    for i, fold in enumerate(_consensus_log, 1):
        print(f"  Fold {i:2d}: entry={fold.get('entry_threshold')}  "
              f"exit={fold.get('exit_threshold')}  "
              f"vb={fold.get('ventana_betas')}  "
              f"vz={fold.get('ventana_zscore')}  "
              f"sizing={'zscore' if fold.get('sizing_by_zscore') else 'equal'}")
    print("═"*52)

    # ── 9. Bloque PARÁMETROS ÓPTIMOS ──────────────────────────────────────────
    print("  PARÁMETROS ÓPTIMOS PARA PAPER (agregación)")
    print("  Regla: moda para discretos, mediana para continuos")
    print("═"*52)
    print(f"  entry_threshold   : {best.get('entry_threshold')}")
    print(f"  exit_threshold    : {best.get('exit_threshold')}")
    print(f"  ventana_betas     : {best.get('ventana_betas')}")
    print(f"  ventana_zscore    : {best.get('ventana_zscore')}")
    print(f"  sizing_by_zscore  : {best.get('sizing_by_zscore')}")
    print("═"*52)

    # ── 10. Stress testing — último fold OOS ─────────────────────────────────
    # Corre sobre un solo fold (grupos n-2 y n-1 como OOS), NO sobre los 5 paths
    # del CPCV. Es un test de robustez puntual a perturbaciones de mercado,
    # no de validación estadística.
    from analysis.stress_testing import (
        StressTester, StressScenario,
        apply_slippage_bps, apply_pnl_drag, scale_pnl,
    )
    from pipeline.splits import make_groups, build_train_segments

    # Stress testing — 4 escenarios sobre el último fold OOS.
    #
    # Decisión de diseño: NO incluimos escenarios de "volatility shock" ni
    # "liquidity shock" porque con precompute los residuos RMT se computan
    # una vez sobre el dataset original. Aplicar oos_transform escala los
    # precios pero NO recomputa la dinámica de factores — el efecto neto
    # se reduce a escalar el PnL. Para un stress de volatilidad real haría
    # falta recalibrar el modelo factorial con datos perturbados, lo cual
    # está fuera del scope del trabajo. Queda como trabajo futuro.
    print("\n[*] Corriendo stress tests sobre el último fold OOS…")
    _groups = make_groups(data, CPCV_CFG.n_groups)
    _n      = CPCV_CFG.n_groups
    _last_combo = (_n - 2, _n - 1)
    _test_set   = set(_last_combo)
    _is_segs    = build_train_segments(
        _groups, _test_set,
        VAL_CFG.label_horizon, VAL_CFG.embargo_pct, VAL_CFG.embargo_bars,
    )
    _oos_data = pl.concat([_groups[i] for i in _last_combo])

    # Runner limpio (sin wrapper de consensus, _RESIDUOS_PD ya está cargado)
    _stress_runner = build_nested_cpcv_runner(
        val_cfg           = VAL_CFG,
        grid              = PARAM_GRID,
        estimator_factory = estimator_factory,
        n_inner_splits    = _N_INNER_SPLITS,
        precompute_fn     = None,
    )

    _scenarios = [
        StressScenario(name="Slippage 5x (50 bps)",  pnl_transform=apply_slippage_bps(50)),
        StressScenario(name="Slippage 10x (100 bps)", pnl_transform=apply_slippage_bps(100)),
        StressScenario(name="Fee drag 5 bps/día",     pnl_transform=apply_pnl_drag(0.0005)),
        StressScenario(name="PnL crush 50%",          pnl_transform=scale_pnl(0.5)),
    ]

    _tester = StressTester(bars_per_year=VAL_CFG.bars_per_year)
    _srep   = _tester.run(_is_segs, _oos_data, _stress_runner, _scenarios)

    def _fmt_stress(met: dict) -> str:
        return (f"Sharpe={met['sharpe']:+.3f}  "
                f"MaxDD={met['max_drawdown']:.2%}  "
                f"AnnRet={met['annualized_return']:.2%}")

    print("\n" + "═"*51)
    print("  STRESS TESTING — último fold OOS")
    print("  (un fold, no los 5 paths CPCV)")
    print("═"*51)
    print(f"  Baseline            : {_fmt_stress(_srep.baseline_metrics)}")
    for _run in _srep.runs:
        _ds = _run.delta_metrics.get("delta_sharpe", float("nan"))
        print(f"  {_run.scenario:<20}: {_fmt_stress(_run.metrics)}  Δ={_ds:+.3f}")
    _wc = _srep.worst_case("sharpe")
    if _wc:
        print(f"\n  Worst case (Sharpe) : {_wc.scenario}  "
              f"Sharpe={_wc.metrics['sharpe']:+.3f}")
    print(f"\n  Nota: stress de volatilidad/correlaciones requiere recalibración")
    print(f"  del modelo factorial RMT — fuera del scope actual.")
    print("═"*51)

    # Serializar para metrics.json
    _stress_out = {
        "note": "4 PnL-side scenarios on the last OOS fold. Vol/liquidity shocks excluded due to precompute architecture (would require RMT model recalibration).",
        "baseline": _srep.baseline_metrics,
        "scenarios": [
            {
                "name":              r.scenario,
                "sharpe":            r.metrics["sharpe"],
                "max_drawdown":      r.metrics["max_drawdown"],
                "annualized_return": r.metrics["annualized_return"],
                "delta_sharpe":      r.delta_metrics.get("delta_sharpe"),
            }
            for r in _srep.runs
        ],
        "worst_case_scenario": _wc.scenario if _wc else None,
    }

    # ── 11. Guardar outputs ───────────────────────────────────────────────────

    # equity_curves.parquet
    rows = []
    for i, ec in enumerate(report.equity_curves, 1):
        for row in ec.iter_rows(named=True):
            rows.append({"path": i, "bar": row["bar"], "equity": row["equity"]})
    ec_df = pd.DataFrame(rows)
    ec_path = _RESULTS_DIR / "equity_curves.parquet"
    ec_df.to_parquet(ec_path)
    print(f"\n[*] Equity curves guardadas en {ec_path}")

    # best_params.json
    best_path = _RESULTS_DIR / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[*] best_params.json guardado en {best_path}")

    # diagnostico_grid (embebido en metrics.json)
    entries = [d["entry_threshold"] for d in _consensus_log]
    sizings = [d["sizing_by_zscore"]  for d in _consensus_log]
    arr = np.array(entries)
    diag = {
        "entry_threshold": {
            "min":    float(arr.min()),
            "p25":    float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75":    float(np.percentile(arr, 75)),
            "max":    float(arr.max()),
            "distribution": {
                "~1.5": sum(1 for v in entries if abs(v - 1.5) < 0.1),
                "~2.0": sum(1 for v in entries if abs(v - 2.0) < 0.1),
                "~2.5": sum(1 for v in entries if abs(v - 2.5) < 0.1),
            },
        },
        "sizing_by_zscore": {
            "True":  sum(sizings),
            "False": len(sizings) - sum(sizings),
        },
    }

    # metrics.json
    metrics_out = {
        "sharpe_mean":        m["sharpe_mean"],
        "sharpe_std":         m["sharpe_std"],
        "sharpes_per_path":   m["sharpes_per_path"],
        "dsr":                m["dsr"],
        "max_drawdown":       m["max_drawdown"],
        "pct_positive_paths": m["pct_positive_paths"],
        "alpha_annualized":   mr.get("alpha"),
        "beta":               mr.get("beta"),
        "turnover_annual":    m.get("turnover_annual_mean"),
        "turnover_per_path":  m.get("turnover_per_path"),
        "n_paths":            m["phi"],
        "n_combos":           m["n_combos"],
        "embargo_bars":       VAL_CFG.embargo_bars,
        "rebalance_frequency": REBALANCE_FREQUENCY,
        "param_grid":         PARAM_GRID,
        "consensus_per_fold": _consensus_log,
        "diagnostico_grid":   diag,
        "stress_testing":     _stress_out,
        "date_range":         [start, end],
        "generated_at":       datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    metrics_path = _RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[*] metrics.json guardado en {metrics_path}")

    # figures/equity_curves.png
    fig_path = _RESULTS_DIR / "figures" / "equity_curves.png"
    _plot_equity_curves(ec_df, fig_path)
    print(f"[*] Plot guardado en {fig_path}")


if __name__ == "__main__":
    main()
