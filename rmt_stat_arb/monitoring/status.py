"""
show_status()   — lectura read-only de daily_state.parquet.
show_universe() — lista los 100 tickers del universo.
No conecta a IBKR, no ejecuta órdenes.
"""
import json
from pathlib import Path

from constants import INITIAL_CAPITAL   # fuente única — ver rmt_stat_arb/constants.py

_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # → rmt_stat_arb/
_STATE_PATH   = _PROJECT_ROOT / "results" / "trading" / "daily_state.parquet"


def show_status() -> None:
    if not _STATE_PATH.exists():
        print("\nSin runs registrados. Correr  python -m rmt_stat_arb paper  primero.")
        return

    try:
        import pandas as pd
        df = pd.read_parquet(_STATE_PATH)
        if df.empty:
            print("\nSin runs registrados. Correr  python -m rmt_stat_arb paper  primero.")
            return

        last = df.iloc[-1]

        current_nav     = float(last["estimated_nav"])
        month_start_nav = float(last["month_start_nav"])
        n_pos           = int(last["n_active_positions"])
        ret_acum        = (current_nav - INITIAL_CAPITAL) / INITIAL_CAPITAL
        drawdown        = (current_nav - month_start_nav) / month_start_nav if month_start_nav else 0.0

        weights: dict = json.loads(last.get("target_weights", "{}") or "{}")
        longs  = {t: w for t, w in weights.items() if w >  1e-6}
        shorts = {t: w for t, w in weights.items() if w < -1e-6}

        print("\n" + "═"*51)
        print("  ESTADO ACTUAL — RMT Stat-Arb")
        print("═"*51)
        print(f"  Último run        : {last['date']}")
        print(f"  Capital actual    : ${current_nav:,.2f}")
        print(f"  Capital inicial   : ${INITIAL_CAPITAL:,.2f}")
        print(f"  Retorno acumulado : {ret_acum:+.2%}")
        print(f"  Drawdown del mes  : {drawdown:+.2%}")
        print(f"  Posiciones activas: {n_pos}")

        if longs:
            longs_str = ", ".join(f"{t} {w:+.2%}" for t, w in sorted(longs.items()))
            print(f"\n  Posiciones long  ({len(longs)}): {longs_str}")
        else:
            print(f"\n  Posiciones long  (0): —")

        if shorts:
            shorts_str = ", ".join(f"{t} {w:+.2%}" for t, w in sorted(shorts.items()))
            print(f"  Posiciones short ({len(shorts)}): {shorts_str}")
        else:
            print(f"  Posiciones short (0): —")

        print("═"*51 + "\n")

    except Exception as e:
        print(f"\n[ERROR] No se pudo leer el estado: {e}")


def show_universe() -> None:
    """Lista los 100 tickers del universo + metadata de datos disponibles."""
    from data.universe import UNIVERSE
    import pandas as pd

    PRICES_PATH = Path(__file__).resolve().parents[1] / "data" / "storage" / "prices.parquet"

    print("\n" + "═" * 51)
    print(f"  UNIVERSO — {len(UNIVERSE)} tickers líquidos del S&P 500")
    print("═" * 51)

    sorted_tickers = sorted(UNIVERSE)
    cols = 10
    for i in range(0, len(sorted_tickers), cols):
        row = sorted_tickers[i:i + cols]
        print("  " + "  ".join(f"{t:<6}" for t in row))

    print("═" * 51)
    print(f"  Fuente            : Yahoo Finance (yfinance)")

    if PRICES_PATH.exists():
        try:
            df = pd.read_parquet(PRICES_PATH)
            start = df.index[0].date()
            end   = df.index[-1].date()
            print(f"  Período disponible: {start} → {end} ({len(df)} días)")
        except Exception:
            print(f"  Período disponible: prices.parquet existe pero no se pudo leer")
    else:
        print(f"  Período disponible: prices.parquet no descargado todavía")

    print(f"  Total tickers     : {len(UNIVERSE)}")
    print("═" * 51 + "\n")
