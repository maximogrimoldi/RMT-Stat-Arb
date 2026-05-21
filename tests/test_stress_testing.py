from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from validation.stress_testing import (
    StressScenario,
    StressTester,
    apply_pnl_drag,
    apply_slippage_bps,
    liquidity_shock,
    scale_pnl,
    volatility_shock,
    shift_numeric_columns,
)


def _make_data(n: int = 60) -> tuple[list[pl.DataFrame], pl.DataFrame]:
    base = date(2015, 1, 1)
    closes = [100.0 + i for i in range(n)]
    opens = [c - 0.5 for c in closes]
    data = pl.DataFrame({
        "timestamp": [base + timedelta(days=i) for i in range(n)],
        "open": opens,
        "close": closes,
    })
    return [data.slice(0, 30), data.slice(30, 15)], data.slice(45, 15)


def _runner(is_segments: list[pl.DataFrame], oos_data: pl.DataFrame):
    is_close = pl.concat(is_segments)["close"].cast(float)
    center = float(is_close.mean())
    rets = oos_data["close"].cast(float) - center
    signals = pl.Series("signals", [1.0] * len(rets))
    return rets.rename("returns"), signals


def test_stress_tester_runs_generic_scenarios():
    is_segments, oos_data = _make_data()
    tester = StressTester(bars_per_year=252)

    report = tester.run(
        is_segments=is_segments,
        oos_data=oos_data,
        runner=_runner,
        scenarios=[
            StressScenario(
                name="shift_close",
                oos_transform=shift_numeric_columns(5.0, columns=["close"]),
            ),
            StressScenario(
                name="half_pnl",
                pnl_transform=scale_pnl(0.5),
            ),
        ],
    )

    assert "sharpe" in report.baseline_metrics
    assert len(report.runs) == 2
    assert report.worst_case("sharpe") is not None


def test_stress_scenario_applies_is_transform():
    is_segments, oos_data = _make_data()
    tester = StressTester(bars_per_year=252)
    scenario = StressScenario(
        name="shift_is",
        is_transform=shift_numeric_columns(10.0, columns=["close"]),
    )

    report = tester.run(is_segments, oos_data, _runner, [scenario])

    assert report.runs[0].scenario == "shift_is"
    assert report.runs[0].delta_metrics["delta_sharpe"] != 0.0


def test_stress_helpers_cover_cost_slippage_volatility_and_liquidity():
    is_segments, oos_data = _make_data()
    tester = StressTester(bars_per_year=252)

    report = tester.run(
        is_segments,
        oos_data,
        _runner,
        [
            StressScenario(name="cost_drag", pnl_transform=apply_pnl_drag(0.25)),
            StressScenario(name="slippage", pnl_transform=apply_slippage_bps(5.0)),
            StressScenario(name="volatility", oos_transform=volatility_shock(1.2, columns=["close"])),
            StressScenario(name="liquidity", oos_transform=liquidity_shock(0.8, columns=["open"])),
        ],
    )

    assert len(report.runs) == 4
    assert report.worst_case("sharpe") is not None
