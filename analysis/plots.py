"""
validation/plots.py — Visualizacion de resultados del backtester.

Funcion publica: plot_equity_vs_benchmark
  Compara la curva de equity de la estrategia contra un indice de referencia.
  Se usa de forma opcional desde ValidationReport.plot_vs_spy().
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_equity_vs_benchmark(
    oos_returns: np.ndarray | list[np.ndarray],
    start_date: str,
    end_date: str,
    strategy_label: str = "Estrategia",
    benchmark_ticker: str = "SPY",
    output: str = "equity_vs_benchmark.png",
    title: str | None = None,
    bars_per_year: int = 52,
) -> None:
    """
    Genera y guarda un grafico comparando la curva de equity de la estrategia
    contra un benchmark descargado automaticamente.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance es requerido para descargar el benchmark. Instalalo con 'pip install yfinance'."
        ) from exc

    fig, ax = plt.subplots(figsize=(14, 6))

    if isinstance(oos_returns, np.ndarray):
        strat_len = len(oos_returns)
        ax.plot(np.cumprod(1 + oos_returns), color="steelblue", linewidth=2, label=strategy_label)
    else:
        for i, arr in enumerate(oos_returns):
            ax.plot(np.cumprod(1 + arr), alpha=0.4, linewidth=1.0, label=f"Path {i + 1}")
        min_len = min(len(a) for a in oos_returns)
        strat_len = min_len
        avg_arr = np.mean(np.stack([a[:min_len] for a in oos_returns]), axis=0)
        ax.plot(np.cumprod(1 + avg_arr), color="black", linewidth=2, label=f"{strategy_label} (promedio)")

    interval = {252: "1d", 52: "1wk", 12: "1mo"}.get(bars_per_year, "1d")
    bench_raw = yf.download(
        benchmark_ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )["Close"]
    bench_prices = bench_raw.dropna().values.flatten()
    bench_rets = np.diff(bench_prices) / bench_prices[:-1]
    bench_rets = bench_rets[:strat_len]
    ax.plot(np.cumprod(1 + bench_rets), color="firebrick", linewidth=2, linestyle="--", label=f"{benchmark_ticker} Buy & Hold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(title or f"{strategy_label} vs. {benchmark_ticker} Buy & Hold ({start_date[:4]}-{end_date[:4]})")
    ax.set_ylabel("Capital normalizado (inicio = 1)")
    freq_label = {252: "dias", 52: "semanas", 12: "meses"}.get(bars_per_year, "barras")
    ax.set_xlabel(f"Barras ({freq_label})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.show()
    print(f"Grafico guardado: {output}")
