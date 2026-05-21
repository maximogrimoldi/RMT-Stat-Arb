from __future__ import annotations
from queue import Empty, Queue

from engine.data_handler import DataHandler
from engine.execution_handler import ExecutionHandler
from engine.events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from engine.portfolio import Portfolio
from strategy.base import Strategy


class EventLoop:

    def __init__(
        self,
        events_queue: Queue,
        data_handler: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution_handler: ExecutionHandler,
    ) -> None:
        self._queue = events_queue
        self._data_handler = data_handler
        self._strategy = strategy
        self._portfolio = portfolio
        self._execution_handler = execution_handler

    def run(self, close_open_positions: bool = True, slippage_pct: float = 0.0) -> None:
        while self._data_handler.has_more_data:
            self._data_handler.update_bars()
            market_event = self._queue.get()

            # Señal del bar anterior → fill al open del bar actual
            self._execution_handler.fill_pending(market_event.open, market_event.timestamp)
            self._drain_queue()  # procesa FillEvents

            # Estrategia ve el close y genera señales para el próximo bar
            self._strategy.on_market(market_event)
            self._portfolio.update_market(market_event)
            self._drain_queue()  # procesa SignalEvents y OrderEvents

        if close_open_positions:
            for symbol in list(self._portfolio._positions):
                self._portfolio.force_close(symbol, slippage_pct=slippage_pct)

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self._queue.get(block=False)
            except Empty:
                break

            if isinstance(event, SignalEvent):
                self._portfolio.on_signal(event)
            elif isinstance(event, OrderEvent):
                self._execution_handler.on_order(event)
            elif isinstance(event, FillEvent):
                self._portfolio.on_fill(event)
