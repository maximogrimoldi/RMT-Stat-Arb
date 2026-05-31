"""
Checks de salud para RMT Stat-Arb paper trading.

Pre-trade : datos frescos, TWS disponible, idempotencia.
Post-trade : capital, drawdown del mes, market-neutral, gross exposure,
             n_posiciones, z-scores extremos.

Lee results/trading/daily_state.parquet.
"""

import json
import socket

import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # → rmt_stat_arb/
_STATE_PATH   = _PROJECT_ROOT / "results" / "trading" / "daily_state.parquet"


def _check_ibkr_ping(host="127.0.0.1", port=7497, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def _load_last_row() -> pd.Series | None:
    """Carga la última fila de daily_state.parquet. Retorna None si no existe o está vacío."""
    if not _STATE_PATH.exists():
        return None
    try:
        df = pd.read_parquet(_STATE_PATH)
        return df.iloc[-1] if not df.empty else None
    except Exception:
        return None


# ── Pre-Trade Checks ──────────────────────────────────────────────────────────

def run_pre_trade_checks(prices_hist, force: bool = False) -> bool:
    """
    3 checks en orden. Retorna False en el primero que falla.
    1. Datos actualizados (último precio == hoy).
    2. TWS disponible en 127.0.0.1:7497.
    3. Idempotencia: date_only de la última fila del parquet != hoy.  (salteado si force=True)
    """
    print("\n" + "═" * 51)
    print("  PRE-TRADE CHECKS")
    print("═" * 51)

    # Check 1: datos frescos
    last_date = prices_hist.index[-1].date()
    today     = pd.Timestamp.today().date()
    if last_date != today:
        print(f"✗ Datos desactualizados — último precio: {last_date}, hoy: {today}")
        print("  Correr descarga antes de ejecutar.")
        print("═" * 51)
        return False
    print(f"✓ Datos frescos: último precio = {last_date}")

    # Check 2: TWS
    if not _check_ibkr_ping():
        print("✗ TWS: sin conexión en 127.0.0.1:7497.")
        print("  Abrí Trader Workstation y volvé a correr.")
        print("═" * 51)
        return False
    print("✓ TWS: responde en 127.0.0.1:7497")

    # Check 3: idempotencia
    if force:
        print("⚠  --force activo: saltando check de idempotencia")
    else:
        today_str = str(today)
        last      = _load_last_row()
        if last is not None and str(last.get("date_only", "")) == today_str:
            print(f"✗ Ya se corrió hoy ({today_str}), no se rebalancea de nuevo.")
            print("  Usar --force para saltear este check.")
            print("═" * 51)
            return False
        print("✓ Idempotencia: no se corrió hoy aún")

    print("═" * 51)
    return True


# ── Post-Trade Health Check ───────────────────────────────────────────────────

def run_health_checks() -> bool:
    """
    Lee la última fila de daily_state.parquet y muestra bloque post-trade:
    capital, drawdown del mes y 4 health checks (warnings, no bloquean).
    """
    print("\n" + "═" * 51)
    print("  POST-TRADE — RMT Stat-Arb")
    print("═" * 51)

    last = _load_last_row()
    if last is None:
        print("✗ No se encontró daily_state.parquet o está vacío.")
        print("═" * 51)
        return False

    try:
        current_nav     = float(last["estimated_nav"])
        month_start_nav = float(last["month_start_nav"])
        n_pos           = int(last["n_active_positions"])
        drawdown        = (current_nav - month_start_nav) / month_start_nav if month_start_nav else 0.0

        print(f"  Capital actual    : ${current_nav:,.2f}")
        print(f"  Drawdown del mes  : {drawdown:+.2%}   (NAV / month_start_nav - 1)")
        print(f"  Posiciones activas: {n_pos}")
        print("═" * 51)
        print("  HEALTH CHECKS")

        target_weights = json.loads(last.get("target_weights", "{}") or "{}")
        zscores_raw    = json.loads(last.get("zscores", "{}") or "{}")
        active_tickers = [t for t, w in target_weights.items() if abs(w) > 1e-6]
        alertas        = 0

        # Market-neutral
        net_exposure = sum(target_weights.values())
        if abs(net_exposure) >= 0.15:
            print(f"  Market-neutral (|Σpesos| < 0.15)  : ⚠  valor={net_exposure:+.3f}")
            alertas += 1
        else:
            print(f"  Market-neutral (|Σpesos| < 0.15)  : ✓  valor={net_exposure:+.3f}")

        # Gross exposure
        gross = sum(abs(w) for w in target_weights.values())
        if not (0.3 < gross < 1.1):
            print(f"  Gross exposure (0.3 < Σ|w| < 1.1) : ⚠  valor={gross:.3f}")
            alertas += 1
        else:
            print(f"  Gross exposure (0.3 < Σ|w| < 1.1) : ✓  valor={gross:.3f}")

        # N posiciones
        if not (5 <= n_pos <= 40):
            print(f"  N posiciones (5 ≤ n ≤ 40)         : ⚠  valor={n_pos}")
            alertas += 1
        else:
            print(f"  N posiciones (5 ≤ n ≤ 40)         : ✓  valor={n_pos}")

        # Z-scores extremos
        if not zscores_raw:
            print("  Z-scores extremos (|z| < 6)        : ⚠  sin datos de z-scores")
        else:
            extremos = [
                (t, zscores_raw[t])
                for t in active_tickers
                if t in zscores_raw
                and zscores_raw[t] is not None
                and abs(zscores_raw[t]) > 6
            ]
            if extremos:
                detalle = ", ".join(f"{t}={z:.2f}" for t, z in extremos)
                print(f"  Z-scores extremos (|z| < 6)        : ⚠  {detalle}")
                alertas += 1
            else:
                print("  Z-scores extremos (|z| < 6)        : ✓  ningún |z|>6 en posiciones activas")

        print("═" * 51)
        if alertas:
            print(f"  ⚠  {alertas} alerta(s). Revisar antes de la próxima sesión.")
        else:
            print("  ✓  Sistema estable.")
        print("═" * 51 + "\n")
        return alertas == 0

    except Exception as e:
        print(f"✗ Falló el monitoreo post-trade: {e}")
        print("═" * 51)
        return False
