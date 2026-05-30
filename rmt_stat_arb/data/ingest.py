"""
Carga y descarga de precios para RMT Stat-Arb.

Diferencias respecto al ingest de DQI:
  - Sin excepción para SPY: RMT no usa regime filter, todos los tickers
    se tratan igual. Si un ticker supera el umbral de NaN, se dropea.
  - PRICES_PATH es absoluto, relativo a la ubicación de este archivo,
    para que no dependa del CWD desde donde se ejecute el script.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# __file__ está en rmt_stat_arb/codigo/data/ingest.py
# parent  → rmt_stat_arb/codigo/data/
_DATA_DIR   = Path(__file__).resolve().parent
PRICES_PATH = _DATA_DIR / "storage" / "prices.parquet"


def download_prices(tickers: list[str], start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Descarga precios ajustados desde Yahoo Finance, limpia y guarda en parquet.

    Limpieza aplicada:
      1. Columna "Close" ajustada (splits + dividendos vía auto_adjust=True).
      2. Descarte de días donde TODOS los tickers son NaN (feriados/fines de semana).
      3. Descarte de tickers con > 20% NaN (datos insuficientes).
      4. Forward-fill del resto de NaN residuales (gaps puntuales).
    """
    print(f"[*] Descargando {len(tickers)} tickers desde {start_date} …")

    data   = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = data["Close"]

    # Descartar días donde TODOS los activos son NaN (feriados / no-trading days)
    prices = prices.dropna(how="all")

    # Descartar tickers con historia demasiado incompleta ANTES del ffill
    nan_pct     = prices.isna().mean()
    bad_tickers = nan_pct[nan_pct > 0.20].index.tolist()
    if bad_tickers:
        print(f"[!] Dropeando {len(bad_tickers)} ticker(s) con >20% NaN: {bad_tickers}")
        prices = prices.drop(columns=bad_tickers)

    # Forward-fill gaps puntuales (ticker sin dato un día puntual)
    prices = prices.ffill()

    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICES_PATH)

    print(f"[*] Guardado en {PRICES_PATH}. Shape: {prices.shape}. "
          f"Última fecha: {prices.index[-1].date()}")
    return prices


def load_prices() -> pd.DataFrame:
    """Carga el parquet de precios guardado por download_prices."""
    return pd.read_parquet(PRICES_PATH)


def check_data_status(tickers: list[str], prices_path: Path = PRICES_PATH) -> bool:
    """
    Devuelve True si el parquet local existe, contiene todos los tickers pedidos
    y está actualizado al último día hábil.
    """
    if not prices_path.exists():
        return False

    saved = pd.read_parquet(prices_path)

    if not set(saved.columns).issuperset(set(tickers)):
        return False

    last_bday = np.busday_offset(
        pd.Timestamp.today().date(), 0, roll="backward"
    ).astype(object)

    return saved.index[-1].date() >= last_bday
