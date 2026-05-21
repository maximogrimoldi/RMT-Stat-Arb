from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from validation.config import ValidationConfig
from validation.tuning import (
    build_nested_cpcv_runner,
    consensus_params,
    tune_flat_dataset,
    tune_grid_on_is,
)


def _make_data(n: int = 180) -> pl.DataFrame:
    base = date(2015, 1, 1)
    closes = [100.0 + i * 0.1 for i in range(n)]
    opens = [c * 1.001 for c in closes]
    return pl.DataFrame({
        "timestamp": [base + timedelta(days=i) for i in range(n)],
        "open": opens,
        "close": closes,
    })


def _runner_factory(params):
    strength = float(params["strength"])

    def runner(is_segments, oos_data):
        n = max(0, len(oos_data) - 1)
        returns = [strength + (0.001 if i % 2 == 0 else -0.001) for i in range(n)]
        signals = [1.0] * n
        return pl.Series("returns", returns), pl.Series("signals", signals)

    return runner


def test_tune_grid_on_is_picks_best_params():
    data = _make_data(90)
    grid = [{"strength": 0.01}, {"strength": 0.02}, {"strength": 0.03}]

    result = tune_grid_on_is(
        data,
        grid,
        _runner_factory,
        n_inner_splits=3,
    )

    assert result.best_candidate.params == {"strength": 0.03}
    assert len(result.candidates) == 3


def test_consensus_params_uses_median_and_mode():
    winners = [
        {"strength": 1, "mode": "slow"},
        {"strength": 3, "mode": "fast"},
        {"strength": 5, "mode": "slow"},
    ]

    consensus = consensus_params(winners)

    assert consensus["strength"] == 3
    assert consensus["mode"] == "slow"


def test_tune_flat_dataset_returns_consensus():
    data = _make_data(180)
    val_cfg = ValidationConfig(label_horizon=1, embargo_pct=0.01)
    grid = [{"strength": 0.01}, {"strength": 0.02}, {"strength": 0.03}]

    result = tune_flat_dataset(
        data,
        val_cfg,
        grid,
        _runner_factory,
        n_splits=5,
    )

    assert result.fold_results
    assert result.consensus_params["strength"] == 0.03


def test_nested_runner_fits_then_predicts_on_oos():
    data = _make_data(90)
    is_segments = [data.slice(0, 30), data.slice(30, 30), data.slice(60, 30)]
    oos_data = _make_data(40)
    val_cfg = ValidationConfig(label_horizon=1, embargo_pct=0.01)
    grid = [{"strength": 0.01}, {"strength": 0.02}, {"strength": 0.03}]
    log: list[tuple] = []

    def estimator_factory(params):
        strength = float(params["strength"])

        class _Model:
            def fit(self, is_segments):
                log.append(("fit", strength, sum(len(seg) for seg in is_segments)))
                return self

            def predict(self, oos_data):
                log.append(("predict", strength, len(oos_data)))
                n = max(0, len(oos_data) - 1)
                returns = [strength] * n
                signals = [strength] * n
                return pl.Series("returns", returns), pl.Series("signals", signals)

        return _Model()

    runner = build_nested_cpcv_runner(
        val_cfg=val_cfg,
        grid=grid,
        estimator_factory=estimator_factory,
        n_inner_splits=3,
    )

    returns, signals = runner(is_segments, oos_data)

    assert len(returns) == len(oos_data) - 1
    assert len(signals) == len(oos_data) - 1
    assert log[-2][0] == "fit"
    assert log[-1][0] == "predict"
    assert log[-1][2] == len(oos_data)
