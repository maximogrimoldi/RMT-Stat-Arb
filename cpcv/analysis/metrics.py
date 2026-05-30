from __future__ import annotations

import numpy as np
import polars as pl
import scipy.stats
from scipy.stats import norm


def sharpe_ratio(returns: pl.Series, annualization_factor: float = 252.0) -> float:
    arr = returns.to_numpy()
    std = arr.std()
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(annualization_factor))


def probabilistic_sharpe_ratio(
    returns: pl.Series,
    benchmark_sr: float = 0.0,
    bars_per_year: float = 252.0,
) -> float:
    """
    PSR = Φ[ (SR̂ − SR*) · √(T−1) / √(1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²) ]
    SR̂ y SR* en escala per-bar. benchmark_sr se recibe anualizado y se convierte.
    γ₄ es excess kurtosis (scipy default), por eso el término es (excess+2)/4.
    """
    arr = returns.to_numpy()
    T = len(arr)
    if T < 4:
        return float("nan")
    std = arr.std(ddof=1)
    if std == 0:
        return float("nan")

    sr = arr.mean() / std
    sr_star = benchmark_sr / np.sqrt(bars_per_year)
    skew = float(scipy.stats.skew(arr))
    kurt = float(scipy.stats.kurtosis(arr))  # excess kurtosis

    variance = (1 - skew * sr + ((kurt + 2) / 4) * sr**2) / (T - 1)
    if variance <= 0:
        return float("nan")

    return float(norm.cdf((sr - sr_star) / np.sqrt(variance)))


def expected_max_sharpe(n_trials: int, sr_mean: float = 0.0, sr_std: float = 1.0) -> float:
    """
    E[max SR] entre n_trials estrategias con SR ~ N(sr_mean, sr_std).
    Aproximación de Bailey & López de Prado (2014).
    sr_mean y sr_std en términos anualizados. Devuelve SR anualizado.
    """
    if n_trials <= 1:
        return sr_mean
    euler = 0.5772156649
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return sr_mean + sr_std * ((1 - euler) * z1 + euler * z2)


def deflated_sharpe_ratio(
    returns: pl.Series,
    n_trials: int,
    benchmark_sr: float = 0.0,
    bars_per_year: float = 252.0,
) -> float:
    """
    DSR: PSR donde SR* se ajusta por el número de trials realizados.
    Prior: SR de estrategias ~ N(0, 1) en términos anualizados.
    """
    sr_star_annualized = expected_max_sharpe(n_trials)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr_star_annualized, bars_per_year=bars_per_year)


def block_bootstrap_sharpe(
    returns: pl.Series,
    n_reps: int = 10_000,
    block_length: int = 20,
    annualization_factor: float = 252.0,
) -> dict[str, float]:
    """
    Bootstrap por bloques (preserva autocorrelación) del Sharpe ratio.
    block_length debería ser igual a la vida media del alpha.
    Retorna {mean, std, p5, p95}.
    """
    arr = returns.to_numpy()
    n = len(arr)
    if n < block_length or n < 4:
        return {"mean": float("nan"), "std": float("nan"), "p5": float("nan"), "p95": float("nan")}

    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    rng = np.random.default_rng(42)

    # Vectorizado: (n_reps, n_blocks) → (n_reps, n_blocks, block_length) → (n_reps, n)
    starts = rng.integers(0, max_start + 1, size=(n_reps, n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_reps, -1)[:, :n]

    synthetic = arr[indices]  # (n_reps, n)
    stds = synthetic.std(axis=1)
    means = synthetic.mean(axis=1)
    sharpes = np.where(stds > 0, means / stds * np.sqrt(annualization_factor), 0.0)

    return {
        "mean": float(sharpes.mean()),
        "std":  float(sharpes.std()),
        "p5":   float(np.percentile(sharpes, 5)),
        "p95":  float(np.percentile(sharpes, 95)),
    }


def annualized_return(returns: pl.Series, bars_per_year: float = 252.0) -> float:
    arr = returns.to_numpy()
    if len(arr) == 0:
        return 0.0
    n_years = len(arr) / bars_per_year
    total = float(np.prod(1 + arr))
    if total <= 0:
        return -1.0
    return float(total ** (1 / n_years) - 1)


def max_drawdown(returns: pl.Series) -> float:
    equity = np.cumprod(1 + returns.to_numpy())
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def market_regression(
    strategy_returns: pl.Series,
    market_returns: pl.Series,
    bars_per_year: float = 252.0,
) -> dict[str, float]:
    """
    OLS de retornos de la estrategia contra el mercado.
    Devuelve alpha anualizado, beta, R² e information ratio (alpha/tracking error).
    """
    y = strategy_returns.to_numpy()
    x = market_returns.to_numpy()
    n = min(len(y), len(x))
    y, x = y[-n:], x[-n:]

    slope, intercept, r_value, _, _ = scipy.stats.linregress(x, y)

    residuals = y - (intercept + slope * x)
    tracking_error = residuals.std()
    alpha_annualized = intercept * bars_per_year
    ir = float(alpha_annualized / (tracking_error * np.sqrt(bars_per_year))) if tracking_error > 0 else 0.0

    return {
        "alpha":              alpha_annualized,
        "beta":               float(slope),
        "r_squared":          float(r_value ** 2),
        "information_ratio":  ir,
    }


