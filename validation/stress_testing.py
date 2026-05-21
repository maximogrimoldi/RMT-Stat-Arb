from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from validation.cpcv import BacktestRunner
from validation.metrics import max_drawdown, sharpe_ratio


DataTransform = Callable[[pl.DataFrame], pl.DataFrame]
PnLTransform = Callable[[pl.Series], pl.Series]


@dataclass(frozen=True)
class StressScenario:
    """
    Escenario agnostico de stress testing.

    Se puede aplicar sobre los segmentos IS, sobre el OOS y/o sobre el PnL.
    Nada asume un modelo concreto.
    """

    name: str
    is_transform: DataTransform | None = None
    oos_transform: DataTransform | None = None
    pnl_transform: PnLTransform | None = None


@dataclass(frozen=True)
class StressRun:
    scenario: str
    returns: pl.Series
    signals: pl.Series
    metrics: dict[str, float]
    delta_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class StressReport:
    baseline_metrics: dict[str, float]
    runs: list[StressRun] = field(default_factory=list)

    def worst_case(self, metric: str = "sharpe") -> StressRun | None:
        if not self.runs:
            return None
        return min(self.runs, key=lambda r: r.metrics.get(metric, float("inf")))


class StressTester:
    """
    Stress testing genérico para cualquier estrategia compatible con BacktestRunner.
    El tester solo conoce:
      1) is_segments,
      2) oos_data,
      3) un runner con el contrato del core.
    """

    def __init__(self, bars_per_year: int = 252) -> None:
        self._bars_per_year = bars_per_year

    def run(
        self,
        is_segments: list[pl.DataFrame],
        oos_data: pl.DataFrame,
        runner: BacktestRunner,
        scenarios: Sequence[StressScenario],
    ) -> StressReport:
        baseline_returns, baseline_signals = runner(is_segments, oos_data)
        baseline_metrics = self._metrics(baseline_returns)
        report = StressReport(baseline_metrics=baseline_metrics)

        for scenario in scenarios:
            stressed_is = [
                scenario.is_transform(seg) if scenario.is_transform is not None else seg
                for seg in is_segments
            ]
            stressed_oos = (
                scenario.oos_transform(oos_data)
                if scenario.oos_transform is not None
                else oos_data
            )

            returns, signals = runner(stressed_is, stressed_oos)
            if scenario.pnl_transform is not None:
                returns = scenario.pnl_transform(returns)

            metrics = self._metrics(returns)
            delta_metrics = {
                f"delta_{key}": metrics[key] - baseline_metrics.get(key, 0.0)
                for key in metrics
            }
            report.runs.append(
                StressRun(
                    scenario=scenario.name,
                    returns=returns,
                    signals=signals,
                    metrics=metrics,
                    delta_metrics=delta_metrics,
                )
            )

        return report

    def _metrics(self, returns: pl.Series) -> dict[str, float]:
        return {
            "sharpe": sharpe_ratio(returns, self._bars_per_year),
            "annualized_return": self._safe_annualized_return(returns),
            "max_drawdown": max_drawdown(returns),
        }

    def _safe_annualized_return(self, returns: pl.Series) -> float:
        arr = returns.to_numpy().astype(float)
        if len(arr) == 0:
            return 0.0

        clipped = np.clip(arr, -0.999999, None)
        log_total = float(np.sum(np.log1p(clipped)))
        n_years = len(arr) / self._bars_per_year
        annual_log = log_total / n_years
        annual_log = float(np.clip(annual_log, -700.0, 700.0))
        return float(np.exp(annual_log) - 1.0)


def shift_numeric_columns(delta: float, columns: list[str] | None = None) -> DataTransform:
    """
    Desplaza columnas numericas por una constante.
    Util para shocks generales de nivel de precios/inputs.
    """

    def transform(df: pl.DataFrame) -> pl.DataFrame:
        cols = columns or [
            name
            for name, dtype in df.schema.items()
            if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
        ]
        exprs = []
        for name in df.columns:
            if name in cols:
                exprs.append((pl.col(name).cast(float) + delta).alias(name))
            else:
                exprs.append(pl.col(name))
        return df.select(exprs)

    return transform


def scale_numeric_columns(factor: float, columns: list[str] | None = None) -> DataTransform:
    """
    Escala columnas numericas por un factor.
    """

    def transform(df: pl.DataFrame) -> pl.DataFrame:
        cols = columns or [
            name
            for name, dtype in df.schema.items()
            if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
        ]
        exprs = []
        for name in df.columns:
            if name in cols:
                exprs.append((pl.col(name).cast(float) * factor).alias(name))
            else:
                exprs.append(pl.col(name))
        return df.select(exprs)

    return transform


def jitter_numeric_columns(
    std: float,
    columns: list[str] | None = None,
    seed: int = 42,
) -> DataTransform:
    """
    Suma ruido gaussiano a columnas numericas.
    """
    rng = np.random.default_rng(seed)

    def transform(df: pl.DataFrame) -> pl.DataFrame:
        cols = columns or [
            name
            for name, dtype in df.schema.items()
            if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64)
        ]
        exprs = []
        for name in df.columns:
            if name in cols:
                noise = rng.normal(0.0, std, size=len(df))
                exprs.append((pl.Series(name, df[name].cast(float).to_numpy() + noise)).alias(name))
            else:
                exprs.append(pl.col(name))
        return df.select(exprs)

    return transform


def scale_pnl(factor: float) -> PnLTransform:
    """
    Escala la serie de retornos final.
    """

    def transform(returns: pl.Series) -> pl.Series:
        return pl.Series(returns.name, returns.to_numpy() * factor)

    return transform


def apply_pnl_drag(drag: float) -> PnLTransform:
    """
    Aplica un drag constante al PnL, util para stress de costos fijos o fee leakage.
    """

    def transform(returns: pl.Series) -> pl.Series:
        return pl.Series(returns.name, returns.to_numpy() - drag)

    return transform


def apply_slippage_bps(bps: float) -> PnLTransform:
    """
    Reduce el PnL de manera simetrica por slippage expresado en basis points.
    """

    drag = abs(bps) / 10_000.0

    def transform(returns: pl.Series) -> pl.Series:
        arr = returns.to_numpy().astype(float)
        adjusted = arr - np.sign(arr) * drag
        return pl.Series(returns.name, adjusted)

    return transform


def volatility_shock(factor: float, columns: list[str] | None = None) -> DataTransform:
    """
    Alias semantico para escalar la amplitud de columnas numericas.
    """
    return scale_numeric_columns(factor, columns=columns)


def liquidity_shock(factor: float, columns: list[str] | None = None) -> DataTransform:
    """
    Alias semantico para estresar liquidez reduciendo o amplificando columnas numericas.
    """
    return scale_numeric_columns(factor, columns=columns)
