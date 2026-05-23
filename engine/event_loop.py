from __future__ import annotations
from queue import Empty, Queue

import polars as pl

from engine.execution_handler import ExecutionHandler
from engine.events import FillEvent, OrderEvent
from engine.portfolio import Portfolio
from strategy.base import Strategy

DATE_COL = "timestamp"


class EventLoop:

    def __init__(
        self,
        events_queue: Queue,
        data: pl.DataFrame,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
    ) -> None:
        self._queue = events_queue
        self._data = data
        self._strategy = strategy
        self._portfolio = portfolio
        self._execution_handler = execution_handler

    def run(self, close_open_positions: bool = True) -> None:
        price_cols = [c for c in self._data.columns if c != DATE_COL]

        for i in range(len(self._data)):
            row       = self._data.row(i, named=True)
            timestamp = row[DATE_COL]
            prices    = {col: row[col] for col in price_cols}
            history   = self._data.slice(0, i + 1)

            # fill órdenes del bar anterior al precio actual
            self._execution_handler.fill_pending(prices, timestamp)
            self._drain_queue()

            # estrategia ve historia hasta el bar actual y devuelve weights
            weights = self._strategy.get_weights(history)

            # portfolio reconcilia posición actual con target weights
            self._portfolio.on_weights(weights, prices, timestamp)
            self._drain_queue()

        if close_open_positions:
            slip = self._execution_handler.slippage_pct
            for symbol in list(self._portfolio._positions):
                self._portfolio.force_close(symbol, slippage_pct=slip)

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self._queue.get(block=False)
            except Empty:
                break

            if isinstance(event, OrderEvent):
                self._execution_handler.on_order(event)
            elif isinstance(event, FillEvent):
                self._portfolio.on_fill(event)
