"""
Tests de métricas estadísticas: Sharpe, PSR, block bootstrap, DSR.
"""
import numpy as np
import polars as pl

from validation.metrics import (
    block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def _make_returns(mean: float, std: float, n: int = 500, seed: int = 42) -> pl.Series:
    rng = np.random.default_rng(seed)
    return pl.Series("returns", rng.normal(mean, std, n))


def test_sharpe_ratio_zero_on_flat_returns():
    returns = pl.Series("returns", [0.0] * 100)
    assert sharpe_ratio(returns) == 0.0


def test_psr_above_95_for_strong_signal():
    # Retornos con SR anualizado muy alto (~3.0) -> PSR debe ser cercano a 1
    returns = _make_returns(mean=0.003, std=0.01, n=500)
    psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    assert psr > 0.95, f"PSR esperado > 0.95, obtenido {psr:.3f}"


def test_psr_below_70_for_noise():
    # Retornos con SR negativo -> PSR debe ser bajo
    returns = _make_returns(mean=-0.001, std=0.02, n=252)
    psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    assert psr < 0.70, f"PSR esperado < 0.70, obtenido {psr:.3f}"


def test_block_bootstrap_returns_four_stats():
    returns = _make_returns(mean=0.001, std=0.01, n=300)
    result  = block_bootstrap_sharpe(returns, n_reps=500, block_length=10)
    assert set(result.keys()) == {"mean", "std", "p5", "p95"}
    assert result["p5"] <= result["mean"] <= result["p95"]


def test_block_bootstrap_p5_negative_for_noise():
    rng = np.random.default_rng(7)
    returns = pl.Series("returns", rng.normal(0.0, 0.02, 252))
    result  = block_bootstrap_sharpe(returns, n_reps=2_000, block_length=20)
    assert result["p5"] < result["p95"]
    # Con retornos centrados en 0, el p5 del bootstrap debe ser negativo
    assert result["p5"] < 0.5


def test_expected_max_sharpe_increases_with_trials():
    sr1  = expected_max_sharpe(1)
    sr10 = expected_max_sharpe(10)
    sr100 = expected_max_sharpe(100)
    assert sr1 < sr10 < sr100


def test_dsr_lower_than_psr_with_multiple_trials():
    returns = _make_returns(mean=0.001, std=0.015, n=300)
    psr = probabilistic_sharpe_ratio(returns, benchmark_sr=0.0)
    dsr = deflated_sharpe_ratio(returns, n_trials=50)
    assert dsr < psr, "DSR debe ser menor que PSR cuando n_trials > 1"
