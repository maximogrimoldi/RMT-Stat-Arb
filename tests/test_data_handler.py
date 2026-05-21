"""
Verifica que la barrera temporal del DataHandler es estricta:
- Nunca entrega datos con timestamp > cursor actual
- El warm-up no contamina el IS
- El IS no puede acceder al OOS
"""
from datetime import datetime, timedelta
from queue import Queue

import polars as pl
import pytest

from engine.data_handler import DataFrameDataHandler


def _make_df(n: int = 10) -> pl.DataFrame:
    base = datetime(2020, 1, 1)
    return pl.DataFrame({
        "timestamp": [base + timedelta(days=i) for i in range(n)],
        "open":  [float(i + 1) for i in range(n)],
        "close": [float(i + 1) for i in range(n)],
    })


def test_temporal_barrier_blocks_future_data():
    """get_latest_bars nunca puede devolver barras del futuro respecto al cursor."""
    q = Queue()
    df = _make_df(10)
    handler = DataFrameDataHandler(q, "X", df)

    # Antes de avanzar el cursor, get_latest_bars devuelve vacío
    result = handler.get_latest_bars("X", n=5)
    assert len(result) == 0

    # Avanzamos 3 barras
    handler.update_bars()
    handler.update_bars()
    handler.update_bars()

    result = handler.get_latest_bars("X", n=5)
    # Debe devolver como máximo las 3 barras ya entregadas
    assert len(result) <= 3
    # El timestamp máximo no puede superar el cursor actual
    max_ts = result["timestamp"].max()
    assert max_ts <= handler.current_timestamp


def test_warmup_not_included_in_is():
    """CSVDataHandler con warmup_bars=N inicia el cursor en N, excluyendo esas barras del IS."""
    from engine.data_handler import CSVDataHandler
    import tempfile, os

    df = _make_df(20)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
        df.write_csv(f)
        fname = f.name
    try:
        q = Queue()
        handler = CSVDataHandler(q, "X", warmup_bars=5)
        handler.load(fname)
        # El cursor comienza en warmup_bars, no en 0
        assert handler._cursor == 5
        # Las primeras barras entregadas corresponden al índice 5 en adelante
        handler.update_bars()
        event = q.get()
        expected_ts = df["timestamp"][5]
        assert event.timestamp == expected_ts
    finally:
        os.unlink(fname)


def test_get_latest_bars_returns_closed_bars_only():
    """get_latest_bars(n=k) devuelve a lo sumo las últimas k barras ya cerradas."""
    q = Queue()
    df = _make_df(10)
    handler = DataFrameDataHandler(q, "X", df)

    for _ in range(5):
        handler.update_bars()

    bars = handler.get_latest_bars("X", n=3)
    assert len(bars) == 3
    # Los closes deben corresponder a las barras 2, 3, 4 (índices base-0)
    assert list(bars["close"]) == [3.0, 4.0, 5.0]


def test_has_more_data_false_when_exhausted():
    """has_more_data es False cuando el cursor alcanza el final del DataFrame."""
    q = Queue()
    df = _make_df(3)
    handler = DataFrameDataHandler(q, "X", df)

    assert handler.has_more_data is True
    handler.update_bars()
    handler.update_bars()
    handler.update_bars()
    assert handler.has_more_data is False
