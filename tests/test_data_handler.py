"""
Verifica que la barrera temporal del DataHandler es estricta:
- Nunca entrega datos con timestamp > cursor actual
- El warm-up no contamina el IS
- El IS no puede acceder al OOS
"""
from datetime import datetime, timedelta

import polars as pl

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
    df = _make_df(10)
    handler = DataFrameDataHandler("X", df)

    result = handler.get_latest_bars("X", n=5)
    assert len(result) == 0

    handler.update_bars()
    handler.update_bars()
    handler.update_bars()

    result = handler.get_latest_bars("X", n=5)
    assert len(result) <= 3
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
        handler = CSVDataHandler("X", warmup_bars=5)
        handler.load(fname)
        assert handler._cursor == 5
        handler.update_bars()
        assert handler.current_bar.timestamp == df["timestamp"][5]
    finally:
        os.unlink(fname)


def test_get_latest_bars_returns_closed_bars_only():
    """get_latest_bars(n=k) devuelve a lo sumo las últimas k barras ya cerradas."""
    df = _make_df(10)
    handler = DataFrameDataHandler("X", df)

    for _ in range(5):
        handler.update_bars()

    bars = handler.get_latest_bars("X", n=3)
    assert len(bars) == 3
    assert list(bars["close"]) == [3.0, 4.0, 5.0]


def test_has_more_data_false_when_exhausted():
    """has_more_data es False cuando el cursor alcanza el final del DataFrame."""
    df = _make_df(3)
    handler = DataFrameDataHandler("X", df)

    assert handler.has_more_data is True
    handler.update_bars()
    handler.update_bars()
    handler.update_bars()
    assert handler.has_more_data is False


def test_get_history_returns_timestamp_and_close():
    """get_history devuelve [timestamp, close] hasta el cursor inclusive."""
    df = _make_df(5)
    handler = DataFrameDataHandler("X", df)

    handler.update_bars()
    handler.update_bars()
    handler.update_bars()

    history = handler.get_history()
    assert history.columns == ["timestamp", "close"]
    assert len(history) == 3
    assert list(history["close"]) == [1.0, 2.0, 3.0]


def test_current_bar_has_open_and_close():
    """current_bar expone open, close y timestamp del bar actual."""
    df = _make_df(5)
    handler = DataFrameDataHandler("X", df)

    assert handler.current_bar is None
    handler.update_bars()
    bar = handler.current_bar
    assert bar.open == 1.0
    assert bar.close == 1.0
