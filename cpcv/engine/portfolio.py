from __future__ import annotations
from queue import Queue

import polars as pl

from engine.events import FillEvent, OrderEvent


class Portfolio:
    """
    Reconcilia target weights con posición actual y genera OrderEvents.
    """

    def __init__(self, events_queue: Queue, initial_capital: float) -> None:
        self._events_queue = events_queue
        self._initial_capital = float(initial_capital)
        self._cash = float(initial_capital)
        self._positions: dict[str, float] = {}
        self._latest_prices: dict[str, float] = {}
        self._equity_curve: list[float] = []
        self._turnover_acum: float = 0.0
        self._last_target_weights: dict[str, float] = {}

    def on_weights(self, weights: dict[str, float], prices: dict[str, float], timestamp) -> None:
        self._latest_prices.update(prices)
        self._equity_curve.append(self.equity)

        delta_total = sum(
            abs(weights.get(t, 0.0) - self._last_target_weights.get(t, 0.0))
            for t in set(weights) | set(self._last_target_weights)
        )
        self._turnover_acum += delta_total
        self._last_target_weights = dict(weights)

        for symbol, target_weight in weights.items():
            price = prices.get(symbol, 0.0)
            if price == 0.0:
                continue

            current_qty   = self._positions.get(symbol, 0.0)
            target_value  = self.equity * target_weight
            current_value = current_qty * price
            delta_qty     = (target_value - current_value) / price

            if abs(delta_qty) < 1e-6:
                continue

            if target_weight == 0.0:
                if current_qty == 0.0:
                    continue
                direction = "EXIT_LONG" if current_qty > 0 else "EXIT_SHORT"
                quantity  = abs(current_qty)
            elif delta_qty > 0:
                direction = "LONG"
                quantity  = delta_qty
            else:
                direction = "SHORT"
                quantity  = abs(delta_qty)

            self._events_queue.put(OrderEvent(
                timestamp=timestamp,
                symbol=symbol,
                quantity=quantity,
                direction=direction,
            ))

    def on_fill(self, event: FillEvent) -> None:
        notional = event.fill_price * event.quantity
        if event.direction in ("LONG", "EXIT_SHORT"):
            self._cash -= notional + event.commission
        else:
            self._cash += notional - event.commission

        if event.direction in ("EXIT_LONG", "EXIT_SHORT"):
            self._positions[event.symbol] = 0
        elif event.direction == "LONG":
            self._positions[event.symbol] = self._positions.get(event.symbol, 0) + event.quantity
        else:  # SHORT
            self._positions[event.symbol] = self._positions.get(event.symbol, 0) - event.quantity

    def update_market(self, prices: dict[str, float], timestamp) -> None:
        """Actualiza precios y equity en días sin rebalanceo."""
        self._latest_prices.update(prices)
        self._equity_curve.append(self.equity)

    def force_close(self, symbol: str, slippage_pct: float = 0.0) -> None:
        qty = self._positions.get(symbol, 0)
        if qty == 0:
            return
        price = self._latest_prices.get(symbol, 0)
        if price == 0:
            return
        slippage   = price * slippage_pct
        fill_price = (price - slippage) if qty > 0 else (price + slippage)
        self._cash += qty * fill_price
        self._positions[symbol] = 0
        if self._equity_curve:
            self._equity_curve[-1] = self.equity

    @property
    def turnover_acum(self) -> float:
        return self._turnover_acum

    @property
    def equity(self) -> float:
        holdings = sum(qty * self._latest_prices.get(sym, 0) for sym, qty in self._positions.items())
        return self._cash + holdings

    @property
    def returns_series(self) -> pl.Series:
        if len(self._equity_curve) < 2:
            return pl.Series("returns", [])
        eq = pl.Series("equity", self._equity_curve)
        return eq.pct_change().drop_nulls().rename("returns")
