from __future__ import annotations
from queue import Queue

import polars as pl

from engine.events import FillEvent, MarketEvent, OrderEvent, OrderDirection, SignalEvent
from strategy.sizing import PositionSizer, FixedFractionSizer


class Portfolio:
    """
    Aplica position sizing y filtros de riesgo.
    Transforma SignalEvents en OrderEvents. No conoce la estrategia ni el sizer.
    """

    def __init__(self, events_queue: Queue, initial_capital: float, sizer: PositionSizer) -> None:
        self._events_queue = events_queue
        self._initial_capital = float(initial_capital)
        self._cash = float(initial_capital)
        self._sizer = sizer
        self._positions: dict[str, float] = {}       # symbol -> cantidad de acciones
        self._latest_prices: dict[str, float] = {}   # symbol -> último close
        self._equity_curve: list[float] = []

    def on_signal(self, event: SignalEvent) -> None:
        price = self._latest_prices.get(event.symbol)
        if price is None or price == 0:
            return

        if event.direction == "EXIT":
            current = self._positions.get(event.symbol, 0)
            if current > 0:
                order_direction: OrderDirection = "EXIT_LONG"
            elif current < 0:
                order_direction = "EXIT_SHORT"
            else:
                return
            quantity = abs(current)
        else:
            order_direction = event.direction
            quantity = self._sizer.size(event, self.equity, price, self._positions)

        if quantity <= 0:
            return

        self._events_queue.put(OrderEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            order_type="MKT",
            quantity=quantity,
            direction=order_direction,
        ))

    def on_fill(self, event: FillEvent) -> None:
        notional = event.fill_price * event.quantity
        if event.direction in ("LONG", "EXIT_SHORT"):   # compra
            self._cash -= notional + event.commission
        else:                                            # SHORT o EXIT_LONG (venta)
            self._cash += notional - event.commission

        if event.direction in ("EXIT_LONG", "EXIT_SHORT"):
            self._positions[event.symbol] = 0
        elif event.direction == "LONG":
            self._positions[event.symbol] = self._positions.get(event.symbol, 0) + event.quantity
        else:  # SHORT
            self._positions[event.symbol] = self._positions.get(event.symbol, 0) - event.quantity

    def update_market(self, event: MarketEvent) -> None:
        self._latest_prices[event.symbol] = event.close
        self._equity_curve.append(self.equity)

    @property
    def equity(self) -> float:
        holdings = sum(qty * self._latest_prices.get(sym, 0) for sym, qty in self._positions.items())
        return self._cash + holdings

    def force_close(self, symbol: str, slippage_pct: float = 0.0) -> None:
        """Cierra la posición abierta al último precio conocido con slippage.
        Llamar al final del fold para no dejar posiciones sin realizar."""
        qty = self._positions.get(symbol, 0)
        if qty == 0:
            return
        price = self._latest_prices.get(symbol, 0)
        if price == 0:
            return
        slippage = price * slippage_pct
        fill_price = (price - slippage) if qty > 0 else (price + slippage)
        self._cash += qty * fill_price
        self._positions[symbol] = 0
        if self._equity_curve:
            self._equity_curve[-1] = self.equity

    @property
    def returns_series(self) -> pl.Series:
        if len(self._equity_curve) < 2:
            return pl.Series("returns", [])
        eq = pl.Series("equity", self._equity_curve)
        return eq.pct_change().drop_nulls().rename("returns")


class SimplePortfolio(Portfolio):
    """Conveniencia: Portfolio con FixedFractionSizer. API idéntica a antes."""

    def __init__(self, events_queue: Queue, initial_capital: float, position_pct: float = 0.02) -> None:
        super().__init__(events_queue, initial_capital, sizer=FixedFractionSizer(position_pct))
