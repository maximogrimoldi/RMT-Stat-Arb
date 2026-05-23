from __future__ import annotations
from queue import Empty, Queue

from engine.data_handler import DataHandler
from engine.execution_handler import ExecutionHandler
from engine.events import FillEvent, OrderEvent
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

    def run(self, close_open_positions: bool = True) -> None:
        while self._data_handler.has_more_data:
            self._data_handler.update_bars()
            bar = self._data_handler.current_bar

            # Señal del bar anterior → fill al open del bar actual
            self._execution_handler.fill_pending(bar.open, bar.timestamp)
            self._drain_queue()  # procesa FillEvents

            # Estrategia ve la historia hasta el close actual y devuelve weights
            history = self._data_handler.get_history()
            weights = self._strategy.get_weights(history)

            # Portfolio reconcilia posición actual con target weights
            self._portfolio.on_weights(weights, bar.close, bar.timestamp)
            self._drain_queue()  # procesa OrderEvents

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
