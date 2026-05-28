"""
Checks de salud para RMT Stat-Arb paper trading.

Pre-trade : conexión IBKR, estimación NAV, stop loss mensual.
Post-trade : estructura del log, market-neutral, gross exposure,
             n_posiciones, z-scores extremos.

No importa signals.py ni core.py — los z-scores se leen del parquet.
"""

import json
import socket

import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # → rmt_stat_arb/
_STATE_PATH   = _PROJECT_ROOT / "results" / "paper" / "daily_state.parquet"
_ORDERS_PATH  = _PROJECT_ROOT / "results" / "paper" / "orders_log.parquet"


# ── NAV Estimation ────────────────────────────────────────────────────────────

def estimate_nav(prices_hist, initial_capital):
    """
    Estima NAV actual y month_start_nav desde el último estado guardado
    más retornos de precios desde esa fecha hasta hoy.
    Devuelve (estimated_nav, month_start_nav).
    """
    if not _STATE_PATH.exists():
        return initial_capital, initial_capital

    state_df = pd.read_parquet(_STATE_PATH)
    if state_df.empty:
        return initial_capital, initial_capital

    last     = state_df.iloc[-1]
    last_nav = float(last["estimated_nav"])

    last_weights_raw = json.loads(last["actual_weights"])
    if not last_weights_raw:
        last_weights_raw = json.loads(last["target_weights"])
    last_weights = pd.Series(last_weights_raw).reindex(prices_hist.columns).fillna(0.0)

    try:
        prices_since = prices_hist.loc[str(last["date"])[:10]:]
    except KeyError:
        estimated_nav = last_nav
    else:
        if len(prices_since) < 2:
            estimated_nav = last_nav
        else:
            daily_rets     = prices_since.pct_change().iloc[1:].fillna(0.0)
            portfolio_rets = daily_rets.dot(last_weights)
            estimated_nav  = last_nav * float((1 + portfolio_rets).prod())

    current_month = pd.Timestamp.today().strftime("%Y-%m")
    month_mask    = state_df["date"].str.startswith(current_month)
    if month_mask.any():
        month_start_nav = float(state_df[month_mask].iloc[0]["estimated_nav"])
    else:
        month_start_nav = estimated_nav

    return estimated_nav, month_start_nav


# ── Helpers internos ──────────────────────────────────────────────────────────

def _check_ibkr_ping(host="127.0.0.1", port=7497, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError):
        return False


# ── Pre-Trade Checks ──────────────────────────────────────────────────────────

def run_pre_trade_checks(prices_hist, initial_capital, stop_loss_pct=0.15):
    print("\n" + "=" * 55)
    print("  PRE-TRADE HEALTH CHECK")
    print("=" * 55)

    # --- Check 1: TWS disponible ---
    if not _check_ibkr_ping():
        print("✗ TWS: sin conexión en 127.0.0.1:7497.")
        print("       Abrí Trader Workstation y volvé a correr.")
        print("=" * 55)
        return False, False, None, None

    print("✓ TWS: responde en 127.0.0.1:7497")

    # --- Check 2: Stop Loss mensual ---
    estimated_nav, month_start_nav = estimate_nav(prices_hist, initial_capital)
    drawdown = (estimated_nav - month_start_nav) / month_start_nav
    nav_line = (
        f"NAV ~${estimated_nav:,.0f} | "
        f"Inicio de mes ~${month_start_nav:,.0f} | "
        f"Drawdown: {drawdown:+.1%}"
    )

    sl_triggered = drawdown < -stop_loss_pct
    if sl_triggered:
        print(f"\n{'!' * 55}")
        print(f"✗ STOP LOSS DISPARADO — {nav_line}")
        print(f"{'!' * 55}")
    else:
        print(f"✓ Stop Loss: dentro del límite — {nav_line}")

    print("=" * 55)
    return True, sl_triggered, estimated_nav, month_start_nav


# ── Post-Trade Health Check ───────────────────────────────────────────────────

def run_health_checks():
    print("\n" + "=" * 55)
    print("  POST-TRADE HEALTH CHECK")
    print("=" * 55)

    if not _STATE_PATH.exists():
        print("✗ No se encontró daily_state.parquet.")
        return False

    try:
        state_df = pd.read_parquet(_STATE_PATH)
        if state_df.empty:
            print("⚠  daily_state.parquet existe pero está vacío — sin registros para chequear.")
            return False
        last     = state_df.iloc[-1]

        last_date = str(last["date"])[:10]
        hoy       = pd.Timestamp.now().strftime("%Y-%m-%d")

        if last_date != hoy:
            print(f"✗ Estado no guardado hoy. Último registro: {last_date}")
            print("=" * 55 + "\n")
            return False

        nav         = float(last["estimated_nav"])
        month_start = float(last["month_start_nav"])
        drawdown    = (nav - month_start) / month_start
        n_pos       = int(last["n_active_positions"])

        print(f"  Registro guardado : {last['date']}")
        print(f"  NAV final         : ${nav:,.2f}")
        print(f"  Drawdown del mes  : {drawdown:+.1%}")
        print(f"  Posiciones activas: {n_pos}")

        if _ORDERS_PATH.exists():
            orders_df = pd.read_parquet(_ORDERS_PATH)
            n_orders  = orders_df["date"].str.contains(hoy, na=False).sum()
            print(f"  Órdenes enviadas  : {n_orders}")

        alertas = 0

        # ── Check 1: drawdown cercano al stop loss ────────────────────────────
        if drawdown < -0.10:
            print(f"\n⚠  Drawdown {drawdown:+.1%} este mes — cerca del Stop Loss.")
            alertas += 1
        else:
            print(f"✓ Drawdown del mes: {drawdown:+.1%}")

        # ── Leer pesos y z-scores ─────────────────────────────────────────────
        target_weights = json.loads(last["target_weights"])

        try:
            zscores_raw = json.loads(last.get("zscores", "{}") or "{}")
        except (TypeError, json.JSONDecodeError):
            zscores_raw = {}

        active_tickers = [t for t, w in target_weights.items() if abs(w) > 1e-6]

        # ── Check 2: Market-neutral ───────────────────────────────────────────
        net_exposure = sum(target_weights.values())
        if abs(net_exposure) >= 0.15:
            print(f"⚠  Market-neutral: exposición neta = {net_exposure:+.3f}  (límite |0.15|)")
            alertas += 1
        else:
            print(f"✓ Market-neutral: exposición neta = {net_exposure:+.3f}")

        # ── Check 3: Gross exposure ───────────────────────────────────────────
        gross = sum(abs(w) for w in target_weights.values())
        if not (0.3 < gross < 1.1):
            print(f"⚠  Gross exposure: {gross:.3f}  (esperado 0.3 < gross < 1.1)")
            alertas += 1
        else:
            print(f"✓ Gross exposure: {gross:.3f}")

        # ── Check 4: N posiciones ─────────────────────────────────────────────
        if not (5 <= n_pos <= 40):
            print(f"⚠  N posiciones: {n_pos}  (esperado 5 ≤ n ≤ 40)")
            alertas += 1
        else:
            print(f"✓ N posiciones: {n_pos}")

        # ── Check 5: Z-scores extremos en posiciones abiertas ─────────────────
        if not zscores_raw:
            print("⚠  Z-scores: columna vacía en el registro — check salteado.")
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
                print(f"⚠  Z-scores extremos (|z|>6) en posiciones abiertas: {detalle}")
                alertas += 1
            else:
                print("✓ Z-scores: ningún |z| > 6 en posiciones abiertas")

        print()
        if alertas == 0:
            print("✓ Ejecución y guardado exitosos. Sistema estable.")
        else:
            print(f"⚠  {alertas} alerta(s) activa(s). Revisar antes de la próxima sesión.")

        print("=" * 55 + "\n")
        return alertas == 0

    except Exception as e:
        print(f"✗ Falló el monitoreo post-trade: {e}")
        return False
