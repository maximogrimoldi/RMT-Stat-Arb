"""
Orquestador de paper trading para RMT Stat-Arb.

Flujo de 7 pasos:
  1. Carga / actualización de datos (100 acciones S&P).
  2. Pre-trade checks: TWS, NAV, stop loss mensual.
  3. Detectar tipo de día: LIQUIDATE / REBALANCE / HOLD.
  4. Si stop loss: confirmación explícita escribiendo CONFIRMAR.
  5. Preview de pesos objetivo + confirmación humana (ENTER / CTRL-C).
  6. Ejecución en IBKR.
  7. Post-trade health check.

Uso:
  python scripts/run_paper.py              # día normal
  python scripts/run_paper.py --bootstrap  # forzar rebalanceo (primer run)
"""

import sys
from pathlib import Path

# ── sys.path: este archivo vive en rmt_stat_arb/scripts/
# Necesita rmt_stat_arb/codigo/ para strategy/engines/data
# y rmt_stat_arb/ para monitoring.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # → rmt_stat_arb/
_CODIGO_DIR   = _PROJECT_ROOT / "codigo"
for _p in [str(_CODIGO_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from data.ingest          import load_prices, download_prices, check_data_status
from data.universe        import UNIVERSE
from strategy.core        import RMTStrategy
from engines.paper_engine import PaperEngine, _save_monitoring_only_log
from monitoring.checks    import run_pre_trade_checks, run_health_checks


def run_paper_trading():
    print("=== Iniciando Sistema de Producción RMT Stat-Arb (Paper Trading) ===")

    # ── Parámetros ────────────────────────────────────────────────────────────
    # Actualizar con consensus_params del último run_validation.py.
    entry_threshold  = 2.0
    exit_threshold   = 1.0
    ventana_betas    = 252
    ventana_zscore   = 252
    sizing_by_zscore = False
    initial_capital  = 100_000.0
    stop_loss_pct    = 0.15

    # ── 1. Carga / Actualización de datos ────────────────────────────────────
    if check_data_status(UNIVERSE):
        print("[*] Usando datos locales actualizados a hoy.")
        prices = load_prices()[UNIVERSE]
    else:
        print("[!] Descargando mercado en vivo desde Yahoo Finance …")
        prices = download_prices(UNIVERSE, "2015-01-01")

    prices    = prices.dropna(how="all")
    last_date = prices.index[-1].date()
    today     = pd.Timestamp.today().date()

    if last_date != today:
        print(f"[ERROR CRÍTICO] Stale data — último precio: {last_date}, hoy: {today}.")
        print("[!] Falla en el proveedor de datos. Abortando.")
        return

    # ── 2. Pre-trade checks ───────────────────────────────────────────────────
    ok, sl_triggered, estimated_nav, month_start_nav = run_pre_trade_checks(
        prices, initial_capital, stop_loss_pct=stop_loss_pct
    )

    if not ok:
        return

    # ── 3. Detectar tipo de día ───────────────────────────────────────────────
    is_bootstrap = "--bootstrap" in sys.argv
    fecha_hoy    = prices.index[-1]
    fecha_ayer   = prices.index[-2]
    is_rebalance = is_bootstrap or (fecha_hoy.month != fecha_ayer.month)

    if sl_triggered:
        status = "LIQUIDATE - Stop Loss (NAV cayó > 15%)"
    elif is_bootstrap:
        status = "REBALANCE - Bootstrap (run inicial)"
    elif is_rebalance:
        status = "REBALANCE - Primer día hábil del mes"
    else:
        status = "HOLD - Día normal (Checks OK)"
        print(f"\n[!] No es día de rebalanceo.")
        print(f"    Último precio disponible : {fecha_hoy.date()}")
        print(f"    Mes actual               : {fecha_hoy.month_name()}")
        print(f"    Status                   : {status}")
        _save_monitoring_only_log(estimated_nav, month_start_nav, prices, status)
        run_health_checks()
        return

    # ── 4. Confirmación explícita antes de liquidar ───────────────────────────
    if sl_triggered:
        caida = (estimated_nav - month_start_nav) / month_start_nav
        print("\n" + "!" * 60)
        print("!!  STOP LOSS DISPARADO                                   !!")
        print(f"!!  NAV estimado : ${estimated_nav:,.0f}                        !!")
        print(f"!!  Inicio de mes: ${month_start_nav:,.0f}  Caída: {caida:.1%}         !!")
        print("!!  Se cerrarán TODAS las posiciones abiertas.            !!")
        print("!" * 60)
        resp = input(
            "\nEscribí CONFIRMAR para liquidar todas las posiciones. "
            "Sin confirmación el sistema no hace nada: "
        ).strip().upper()
        if resp != "CONFIRMAR":
            print("Abortando. No se ejecutó ninguna orden.")
            return

    # ── 5. Estrategia + preview ───────────────────────────────────────────────
    strategy = RMTStrategy(
        entry_threshold  = entry_threshold,
        exit_threshold   = exit_threshold,
        ventana_betas    = ventana_betas,
        ventana_zscore   = ventana_zscore,
        sizing_by_zscore = sizing_by_zscore,
    )
    engine = PaperEngine(strategy)

    if not sl_triggered:
        target_weights, _ = engine.compute_target_weights(prices)

        longs  = {t: w for t, w in target_weights.items() if w >  0.001}
        shorts = {t: w for t, w in target_weights.items() if w < -0.001}

        print("\n--- PESOS OBJETIVO ---")
        if longs:
            print("  Longs  (" + str(len(longs)) + "): " +
                  ", ".join(f"{t} {w:+.1%}" for t, w in sorted(longs.items())))
        else:
            print("  Longs  (0): —")
        if shorts:
            print("  Shorts (" + str(len(shorts)) + "): " +
                  ", ".join(f"{t} {w:+.1%}" for t, w in sorted(shorts.items())))
        else:
            print("  Shorts (0): —")
        print(f"  Net exposure : {sum(target_weights.values()):+.3f}")
        print(f"  Gross        : {sum(abs(w) for w in target_weights.values()):.3f}")
        print(f"  Posiciones   : {len(longs) + len(shorts)}")

        input("\n[?] ENTER para ejecutar las órdenes en IBKR, CTRL+C para abortar...")

    # ── 6. Ejecución ──────────────────────────────────────────────────────────
    engine.execute(
        prices,
        is_bootstrap    = is_bootstrap,
        force_liquidate = sl_triggered,
        estimated_nav   = estimated_nav,
        month_start_nav = month_start_nav,
        status          = status,
    )

    # ── 7. Post-trade health check ────────────────────────────────────────────
    run_health_checks()


if __name__ == "__main__":
    run_paper_trading()
