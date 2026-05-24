from __future__ import annotations

from queue import Queue
from typing import Any, Callable

import polars as pl

from engine.execution_handler import SimulatedExecutionHandler
from engine.event_loop import EventLoop
from engine.portfolio import Portfolio
from strategy.base import Strategy


class EventDrivenEstimator:
    """
    Envuelve una Strategy event-driven como FitPredictEstimator.

    Permite que build_nested_cpcv_runner use el engine completo en cada
    evaluación del tuning: slippage, comisiones y fills realistas.
    """

    def __init__(
        self,
        strategy_factory: Callable[..., Strategy],
        params: dict[str, Any],
        initial_capital: float,
        slippage_pct: float = 0.0,
        derecho_mercado_pct: float = 0.0,
        arancel_alyc_pct: float = 0.0,
        rebalance_frequency: str = "daily",
    ) -> None:
        self._strategy_factory = strategy_factory
        self._params = params
        self._initial_capital = initial_capital
        self._slippage_pct = slippage_pct
        self._derecho_mercado_pct = derecho_mercado_pct
        self._arancel_alyc_pct = arancel_alyc_pct
        self._rebalance_frequency = rebalance_frequency
        self._is_segments: list[pl.DataFrame] = []

    def fit(self, is_segments: list[pl.DataFrame]) -> "EventDrivenEstimator":
        self._is_segments = is_segments
        return self

    def predict(self, oos_data: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        strategy = self._strategy_factory(**self._params)
        if hasattr(strategy, "fit"):
            strategy.fit(self._is_segments)

        queue     = Queue()
        portfolio = Portfolio(queue, self._initial_capital)
        execution = SimulatedExecutionHandler(
            queue,
            slippage_pct        = self._slippage_pct,
            derecho_mercado_pct = self._derecho_mercado_pct,
            arancel_alyc_pct    = self._arancel_alyc_pct,
        )

        loop = EventLoop(queue, oos_data, strategy, portfolio, execution,
                         rebalance_frequency=self._rebalance_frequency)
        loop.run()

        returns = portfolio.returns_series
        signals = pl.Series("signals", [1.0] * len(returns))
        return returns, signals
