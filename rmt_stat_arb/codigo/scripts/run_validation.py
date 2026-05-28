"""
Orquestador de la validación CPCV para la estrategia RMT stat-arb.

Flujo:
  1. Carga precios reales (descarga si no están o están desactualizados).
  2. Llama a run_cpcv con la estrategia y su grilla de parámetros.
  3. Guarda métricas en results/metrics/ (json + csv).
  4. Guarda equity curves en results/metrics/ (parquet + csv).
  5. Plotea los paths en results/figures/.
  6. Imprime resumen en consola.

Para conectar el backtester real, cambiar el import de stub_cpcv → cpcv:
    from validation.stub_cpcv import run_cpcv   # ← stub temporal
    from validation.cpcv      import run_cpcv   # ← implementación real
"""

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── sys.path: permite correr el script directamente desde cualquier CWD ──────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.ingest   import load_prices, download_prices, check_data_status
from data.universe import UNIVERSE
from strategy.core import RMTStrategy
from validation.stub_cpcv import run_cpcv   # swap → validation.cpcv cuando esté listo



_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR  = _PROJECT_ROOT / "results"

# ── Parámetros ────────────────────────────────────────────────────────────────
START_DATE   = "2015-01-01"
N_GROUPS     = 10       # grupos CPCV
N_TEST       = 2        # grupos OOS por split
COMMISSIONS  = 0.0001   # 1 bp por operación
# ─────────────────────────────────────────────────────────────────────────────


def load_data():
    """
    Carga precios locales si están actualizados; descarga de Yahoo Finance si no.
    Siempre devuelve un DataFrame limpio con todos los tickers del UNIVERSE
    que pasaron el filtro de calidad (>20% NaN dropeados en ingest).
    """
    if check_data_status(UNIVERSE):
        print("[*] Datos locales actualizados. Cargando desde disco …")
        prices = load_prices()
    else:
        print("[!] Datos desactualizados o faltantes. Descargando desde Yahoo Finance …")
        prices = download_prices(UNIVERSE, start_date=START_DATE)
        print(f"[*] Descarga completa. Última fecha: {prices.index[-1].date()}")

    # Filtrar solo los tickers del UNIVERSE que sobrevivieron el ingest
    available = [t for t in UNIVERSE if t in prices.columns]
    dropped   = [t for t in UNIVERSE if t not in prices.columns]
    if dropped:
        print(f"[!] Tickers no disponibles (dropeados en ingest): {dropped}")

    return prices[available]


def save_metrics(metrics: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # CSV de una sola fila para inspección rápida
    flat = {k: [v] for k, v in metrics.items() if not isinstance(v, dict)}
    flat["consensus_params"] = [json.dumps(metrics.get("consensus_params", {}))]
    pd.DataFrame(flat).to_csv(results_dir / "validation_metrics.csv", index=False)

    print(f"[*] Métricas guardadas en {results_dir}")


def save_equity_curves(equity_curves: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    equity_curves.to_parquet(results_dir / "equity_curves.parquet")
    equity_curves.to_csv(results_dir / "equity_curves.csv", index=False)
    print(f"[*] Equity curves guardadas en {results_dir}")


def validar_resultado(resultado: dict) -> None:
    """
    Verifica que el dict devuelto por run_cpcv tenga la estructura esperada.
    Lanza AssertionError con mensaje claro si algo falta o tiene tipo incorrecto.
    Llamar inmediatamente después de run_cpcv, antes de guardar o plotear.
    """
    assert isinstance(resultado, dict), \
        "run_cpcv debe devolver un dict, recibido: " + type(resultado).__name__

    for clave in ("metrics", "equity_curves"):
        assert clave in resultado, \
            f"Falta '{clave}' en el resultado de run_cpcv. " \
            f"Claves presentes: {list(resultado.keys())}"

    metrics = resultado["metrics"]
    assert isinstance(metrics, dict), \
        f"'metrics' debe ser dict, recibido: {type(metrics).__name__}"

    for clave in ("sharpe_mean", "sharpe_std", "psr", "dsr", "n_paths", "consensus_params"):
        assert clave in metrics, \
            f"Falta '{clave}' en metrics. " \
            f"Claves presentes: {list(metrics.keys())}"

    for clave in ("sharpe_mean", "sharpe_std", "psr", "dsr"):
        assert isinstance(metrics[clave], (int, float)), \
            f"metrics['{clave}'] debe ser numérico, recibido: {type(metrics[clave]).__name__}"

    assert isinstance(metrics["n_paths"], int), \
        f"metrics['n_paths'] debe ser int, recibido: {type(metrics['n_paths']).__name__}"

    assert isinstance(metrics["consensus_params"], dict), \
        f"metrics['consensus_params'] debe ser dict, recibido: {type(metrics['consensus_params']).__name__}"

    ec = resultado["equity_curves"]
    assert isinstance(ec, pd.DataFrame), \
        f"'equity_curves' debe ser pd.DataFrame, recibido: {type(ec).__name__}"
    assert len(ec.columns) > 0, \
        "'equity_curves' no tiene columnas (0 paths)"
    assert len(ec) > 0, \
        "'equity_curves' está vacío (0 filas)"


def _fmt_currency(x: float, _pos=None) -> str:
    """Formatea valores del eje Y como $K / $M sin notación científica."""
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"


def plot_equity_curves(equity_curves: pd.DataFrame, metrics: dict, figures_dir: Path) -> None:
    """
    Plotea las equity curves del resultado CPCV.

    Defensivo respecto al índice: si es DatetimeIndex, formatea como años;
    si es numérico, formatea como días de trading. Funciona con cualquier
    número de paths (columnas del DataFrame).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_paths      = len(equity_curves.columns)
    is_datetime  = isinstance(equity_curves.index, pd.DatetimeIndex)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Paths individuales (semitransparentes) — se adapta a cualquier n_paths
    for col in equity_curves.columns:
        ax.plot(equity_curves.index, equity_curves[col],
                alpha=0.35, linewidth=0.8, color="steelblue")

    # Media de paths (destacada)
    media = equity_curves.mean(axis=1)
    ax.plot(equity_curves.index, media,
            color="navy", linewidth=2.2, label="Media de paths")

    # Banda ±1σ (solo útil con ≥ 2 paths)
    if n_paths >= 2:
        std = equity_curves.std(axis=1)
        ax.fill_between(equity_curves.index, media - std, media + std,
                        alpha=0.12, color="steelblue", label="±1σ entre paths")

    # ── Eje X: fechas si DatetimeIndex, días si numérico ─────────────────────
    if is_datetime:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        xlabel = "Año"
    else:
        xlabel = "Días de trading"

    # ── Eje Y: formato $K / $M ────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_currency))

    # ── Grilla ────────────────────────────────────────────────────────────────
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.4, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.4)

    # ── Título principal ──────────────────────────────────────────────────────
    fig.suptitle(
        "Equity curves CPCV — distribución de trayectorias OOS",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Subtítulo con métricas ────────────────────────────────────────────────
    metrics_line = (
        f"Sharpe: {metrics['sharpe_mean']:.3f} ± {metrics['sharpe_std']:.3f}"
        f"   |   PSR: {metrics['psr']:.2%}"
        f"   |   DSR: {metrics['dsr']:.2%}"
        f"   |   N paths: {metrics['n_paths']}"
    )
    ax.set_title(metrics_line, fontsize=9, color="dimgray", pad=6)

    # ── Leyenda ───────────────────────────────────────────────────────────────
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Equity ($)", fontsize=10)
    fig.tight_layout()

    out = figures_dir / "equity_curves_paths.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Plot guardado en {out}  ({n_paths} paths)")


def print_summary(metrics: dict, n_tickers: int) -> None:
    cp = metrics.get("consensus_params", {})
    print()
    print("═" * 52)
    print("  RESULTADO VALIDACIÓN CPCV — RMT Stat-Arb")
    print("═" * 52)
    print(f"  Universo        : {n_tickers} acciones")
    print(f"  Sharpe medio    : {metrics['sharpe_mean']:.3f}")
    print(f"  Sharpe std      : {metrics['sharpe_std']:.3f}")
    print(f"  PSR             : {metrics['psr']:.2%}")
    print(f"  DSR             : {metrics['dsr']:.2%}")
    print(f"  N paths         : {metrics['n_paths']}")
    print(f"  Params consenso :")
    for k, v in cp.items():
        print(f"      {k}: {v}")
    print("═" * 52)


def main():
    results_metrics = _RESULTS_DIR / "metrics"
    results_figures = _RESULTS_DIR / "figures"

    # 1. Datos reales
    precios = load_data()
    print(f"[*] Universo efectivo: {precios.shape[1]} activos × {len(precios)} días "
          f"({precios.index[0].date()} → {precios.index[-1].date()})")

    # 2. Validación CPCV
    print("[*] Corriendo validación CPCV …")
    resultado = run_cpcv(
        StrategyClass = RMTStrategy,
        data          = precios,
        grid          = RMTStrategy().param_grid(),
        n_groups      = N_GROUPS,
        n_test        = N_TEST,
        commissions   = COMMISSIONS,
    )

    # Validar estructura del resultado antes de usarlo
    validar_resultado(resultado)

    metrics       = resultado["metrics"]
    equity_curves = resultado["equity_curves"]

    # Asignar DatetimeIndex solo si el backtester devolvió índice numérico
    # (caso stub). Si ya viene con DatetimeIndex, no tocar.
    if not isinstance(equity_curves.index, pd.DatetimeIndex):
        equity_curves.index = precios.index[: len(equity_curves)]

    # 3–4. Persistencia
    save_metrics(metrics, results_metrics)
    save_equity_curves(equity_curves, results_metrics)

    # 5. Plot
    plot_equity_curves(equity_curves, metrics, results_figures)

    # 6. Resumen
    print_summary(metrics, n_tickers=precios.shape[1])


if __name__ == "__main__":
    main()
