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
        data: pl.DataFrame | list[pl.DataFrame],
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
        rebalance_frequency: str = "daily",
    ) -> None:
        self._queue = events_queue
        self._segments = [data] if isinstance(data, pl.DataFrame) else list(data)
        self._strategy = strategy
        self._portfolio = portfolio
        self._execution_handler = execution_handler
        self._rebalance_frequency = rebalance_frequency

    def run(self, close_open_positions: bool = True) -> None:
        for i, segment in enumerate(self._segments):
            self._run_segment(segment)
            is_last = (i == len(self._segments) - 1)
            if not is_last:
                slip = self._execution_handler.slippage_pct
                for symbol in list(self._portfolio._positions):
                    self._portfolio.force_close(symbol, slippage_pct=slip)

        if close_open_positions:
            slip = self._execution_handler.slippage_pct
            for symbol in list(self._portfolio._positions):
                self._portfolio.force_close(symbol, slippage_pct=slip)

    def _run_segment(self, segment: pl.DataFrame) -> None:
        price_cols     = [c for c in segment.columns if c != DATE_COL]
        prev_timestamp = None

        for i in range(len(segment)):
            row       = segment.row(i, named=True)
            timestamp = row[DATE_COL]
            prices    = {col: row[col] for col in price_cols}

            self._execution_handler.fill_pending(prices, timestamp)
            self._drain_queue()

            if self._is_rebalance_day(timestamp, prev_timestamp):
                history = segment.slice(0, i + 1)
                weights = self._strategy.get_weights(history)
                self._portfolio.on_weights(weights, prices, timestamp)
                self._drain_queue()
            else:
                self._portfolio.update_market(prices, timestamp)

            prev_timestamp = timestamp

    def _is_rebalance_day(self, timestamp, prev_timestamp) -> bool:
        if self._rebalance_frequency == "daily" or prev_timestamp is None:
            return True
        ts   = timestamp.date()      if hasattr(timestamp, "date")      else timestamp
        prev = prev_timestamp.date() if hasattr(prev_timestamp, "date") else prev_timestamp
        if self._rebalance_frequency == "weekly":
            return (ts.year, ts.isocalendar()[1]) != (prev.year, prev.isocalendar()[1])
        if self._rebalance_frequency == "monthly":
            return (ts.year, ts.month) != (prev.year, prev.month)
        return True

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
