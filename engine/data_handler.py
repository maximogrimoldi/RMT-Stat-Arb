from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from queue import Queue

import polars as pl

from engine.events import MarketEvent


class DataHandler(ABC):
    """
    Guardián del tiempo. Oráculo de solo lectura del pasado.
    Ningún componente puede solicitar datos con timestamp > cursor actual.
    """

    def __init__(self, events_queue: Queue, warmup_bars: int = 0) -> None:
        self._events_queue = events_queue
        self._warmup_bars = warmup_bars
        self._cursor: int = 0
        self._data: pl.DataFrame | None = None

    @abstractmethod
    def load(self, source: str) -> None:
        pass

    @abstractmethod
    def update_bars(self) -> None:
        pass

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> pl.DataFrame:
        pass

    @property
    @abstractmethod
    def current_timestamp(self) -> datetime | None:
        pass

    @property
    @abstractmethod
    def has_more_data(self) -> bool:
        pass


class DataFrameDataHandler(DataHandler):
    """DataHandler que recibe un DataFrame directamente en lugar de un archivo."""

    def __init__(self, events_queue: Queue, symbol: str, data: pl.DataFrame) -> None:
        super().__init__(events_queue)
        self._symbol = symbol
        self._data = data.sort("timestamp")
        self._cursor = 0

    def load(self, source: str) -> None:
        pass

    def update_bars(self) -> None:
        if not self.has_more_data:
            return
        row = self._data.row(self._cursor, named=True)
        self._cursor += 1
        self._events_queue.put(MarketEvent(
            timestamp=row["timestamp"],
            symbol=self._symbol,
            open=row["open"],
            close=row["close"],
        ))

    def get_latest_bars(self, symbol: str, n: int = 1) -> pl.DataFrame:
        start = max(0, self._cursor - n)
        return self._data.slice(start, self._cursor - start)

    @property
    def current_timestamp(self) -> datetime | None:
        if self._cursor == 0:
            return None
        return self._data.row(self._cursor - 1, named=True)["timestamp"]

    @property
    def has_more_data(self) -> bool:
        return self._data is not None and self._cursor < len(self._data)


class CSVDataHandler(DataFrameDataHandler):

    def __init__(self, events_queue: Queue, symbol: str, warmup_bars: int = 0) -> None:
        DataHandler.__init__(self, events_queue, warmup_bars)
        self._symbol = symbol
        self._cursor = warmup_bars

    def load(self, source: str) -> None:
        self._data = pl.read_csv(source, try_parse_dates=True).sort("timestamp")
