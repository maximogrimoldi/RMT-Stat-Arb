from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from queue import Queue

from engine.events import FillEvent, OrderEvent, OrderDirection


class ExecutionHandler(ABC):

    def __init__(self, events_queue: Queue) -> None:
        self._events_queue = events_queue

    @abstractmethod
    def on_order(self, event: OrderEvent) -> None:
        """Encola la orden para ejecutar en el próximo open."""
        pass

    @abstractmethod
    def fill_pending(self, open_price: float, timestamp: datetime) -> None:
        """Ejecuta órdenes pendientes al open del bar actual."""
        pass

    @abstractmethod
    def calculate_slippage(self, price: float, quantity: float) -> float:
        pass

    @abstractmethod
    def calculate_commission(self, quantity: float, fill_price: float) -> float:
        pass


class SimulatedExecutionHandler(ExecutionHandler):

    def __init__(
        self,
        events_queue: Queue,
        slippage_pct: float = 0.0,
        derecho_mercado_pct: float = 0.0,
        arancel_alyc_pct: float = 0.0,
    ) -> None:
        super().__init__(events_queue)
        self._slippage_pct = slippage_pct
        self._derecho_mercado_pct = derecho_mercado_pct
        self._arancel_alyc_pct = arancel_alyc_pct
        self._pending: list[OrderEvent] = []

    def on_order(self, event: OrderEvent) -> None:
        self._pending.append(event)

    def fill_pending(self, open_price: float, timestamp: datetime) -> None:
        for order in self._pending:
            slippage = self.calculate_slippage(open_price, order.quantity)
            if order.direction in ("LONG", "EXIT_SHORT"):  # compra
                fill_price = open_price + slippage
            else:                                           # SHORT o EXIT_LONG (venta)
                fill_price = open_price - slippage
            commission = self.calculate_commission(order.quantity, fill_price)
            self._events_queue.put(FillEvent(
                timestamp=timestamp,
                symbol=order.symbol,
                direction=order.direction,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage,
            ))
        self._pending.clear()

    def calculate_slippage(self, price: float, quantity: float) -> float:
        return price * self._slippage_pct

    def calculate_commission(self, quantity: float, fill_price: float) -> float:
        return quantity * fill_price * (self._derecho_mercado_pct + self._arancel_alyc_pct)
