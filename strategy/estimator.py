from __future__ import annotations

from queue import Queue
from typing import Any, Callable

import polars as pl

from engine.data_handler import DataFrameDataHandler
from engine.execution_handler import SimulatedExecutionHandler
from engine.event_loop import EventLoop
from engine.portfolio import SimplePortfolio
from strategy.base import Strategy


class EventDrivenEstimator:
    """
    Envuelve una Strategy event-driven como FitPredictEstimator.

    Permite que build_nested_cpcv_runner use el engine completo en cada
    evaluación del tuning: slippage, comisiones y fills al open reales.

    Contrato para la Strategy:
      - fit(is_segments)  : estima parámetros sobre IS, no toca buffers rolling
      - reset()           : limpia buffers rolling, preserva parámetros fiteados
      - on_market(event)  : lógica barra a barra sobre OOS
    """

    def __init__(
        self,
        strategy_factory: Callable[..., Strategy],
        params: dict[str, Any],
        symbol: str,
        initial_capital: float,
        position_pct: float,
        slippage_pct: float = 0.0,
        derecho_mercado_pct: float = 0.0,
        arancel_alyc_pct: float = 0.0,
    ) -> None:
        self._strategy_factory = strategy_factory
        self._params = params
        self._symbol = symbol
        self._initial_capital = initial_capital
        self._position_pct = position_pct
        self._slippage_pct = slippage_pct
        self._derecho_mercado_pct = derecho_mercado_pct
        self._arancel_alyc_pct = arancel_alyc_pct
        self._strategy: Strategy | None = None

    def fit(self, is_segments: list[pl.DataFrame]) -> "EventDrivenEstimator":
        queue = Queue()
        strategy = self._strategy_factory(queue, **self._params)
        if hasattr(strategy, "fit"):
            strategy.fit(is_segments)
        self._strategy = strategy
        return self

    def predict(self, oos_data: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        if self._strategy is None:
            raise RuntimeError("Llamar fit() antes de predict().")

        # Limpia buffers rolling pero preserva parámetros fiteados
        self._strategy.reset()

        queue = Queue()
        handler   = DataFrameDataHandler(queue, self._symbol, oos_data)
        portfolio = SimplePortfolio(queue, self._initial_capital, self._position_pct)
        execution = SimulatedExecutionHandler(
            queue,
            slippage_pct        = self._slippage_pct,
            derecho_mercado_pct = self._derecho_mercado_pct,
            arancel_alyc_pct    = self._arancel_alyc_pct,
        )

        # La strategy necesita la nueva queue para emitir eventos
        self._strategy._events_queue = queue

        loop = EventLoop(queue, handler, self._strategy, portfolio, execution)
        loop.run()

        returns = portfolio.returns_series
        signals = pl.Series("signals", [1.0] * len(returns))
        return returns, signals
