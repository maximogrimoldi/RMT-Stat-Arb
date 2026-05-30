"""
Tests del CPCVEngine: geometría de paths, purging/embargo, métricas.
"""
from datetime import date, timedelta

import numpy as np
import polars as pl

from pipeline.config import ValidationConfig
from pipeline.cpcv import CPCVConfig, CPCVEngine


# ── fixtures ────────────────────────────────────────────────────────────────

def _make_data(n: int = 1200) -> pl.DataFrame:
    base = date(2015, 1, 1)
    rng  = np.random.default_rng(42)
    closes = np.cumprod(1 + rng.normal(0.0003, 0.01, n)) * 100.0
    opens  = closes * (1 + rng.normal(0.0, 0.001, n))
    return pl.DataFrame({
        "timestamp": [base + timedelta(days=i) for i in range(n)],
        "open":      opens.tolist(),
        "close":     closes.tolist(),
    })


def _positive_runner(is_segments: list[pl.DataFrame], oos_data: pl.DataFrame):
    rng = np.random.default_rng(1)
    n   = max(0, len(oos_data) - 1)
    return (
        pl.Series("returns", rng.normal(0.002, 0.008, n)),
        pl.Series("signals", [1.0] * n),
    )


def _dummy_runner(is_segments: list[pl.DataFrame], oos_data: pl.DataFrame):
    rng = np.random.default_rng(0)
    n   = max(0, len(oos_data) - 1)
    return (
        pl.Series("returns", rng.normal(0.0, 0.01, n)),
        pl.Series("signals", np.sign(rng.normal(0, 1, n)).tolist()),
    )


def _base_config(**kw) -> ValidationConfig:
    return ValidationConfig(**kw)


def test_cpcv_produces_phi_paths():
    data   = _make_data(1200)
    report = CPCVEngine(_base_config(), CPCVConfig(n_groups=6, n_test_groups=2)).run(data, _positive_runner)

    assert report.metrics["phi"] == 5
    assert len(report.metrics["sharpes_per_path"]) == 5
    assert len(report.equity_curves) == 5


# ── tests de purging y embargo ───────────────────────────────────────────────

def test_train_segments_apply_purge_and_embargo():
    data   = _make_data(600)
    config = _base_config(label_horizon=5, embargo_pct=0.02)
    engine = CPCVEngine(config, CPCVConfig(n_groups=6, n_test_groups=2))
    groups = engine._make_groups(data)

    train_segs    = engine._get_train_segments(groups, {2, 4})
    raw_train_total = sum(len(groups[i]) for i in [0, 1, 3, 5])
    actual_total    = sum(len(s) for s in train_segs)
    assert actual_total < raw_train_total


# ── tests de métricas ────────────────────────────────────────────────────────

def test_cpcv_report_has_metrics_and_dsr_when_requested():
    data   = _make_data(1200)
    report = CPCVEngine(_base_config(n_trials=15), CPCVConfig(n_groups=6, n_test_groups=2)).run(data, _positive_runner)

    assert "sharpe_avg_path" in report.metrics
    assert "psr_avg_path"    in report.metrics
    assert "max_drawdown"    in report.metrics
    assert "dsr" in report.metrics


def test_cpcv_fails_if_k_equals_n():
    import pytest
    data = _make_data(600)
    with pytest.raises(ValueError):
        CPCVEngine(_base_config(), CPCVConfig(n_groups=4, n_test_groups=4)).run(data, _dummy_runner)
