"""
Orquestador de paper trading para RMT Stat-Arb.

Flujo:
  1. Carga / actualización de datos.
  2. Pre-trade checks: datos frescos, TWS, idempotencia.
  3. Leer parámetros validados desde results/best_params.json.
  4. Calcular pesos objetivo + preview.
  5. Confirmación humana (ENTER / CTRL-C).
  6. Ejecución en IBKR.
  7. Post-trade health check.

Uso (vía CLI):
  python -m rmt_stat_arb paper [--force]

Uso directo (legacy):
  python rmt_stat_arb/scripts/run_paper.py [--force]
"""

import json
import sys
from pathlib import Path

# ── sys.path ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # → rmt_stat_arb/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.ingest       import load_prices, download_prices, check_data_status
from data.universe     import UNIVERSE
from strategy.core     import RMTStrategy
from engines.paper_engine import PaperEngine
from monitoring.checks import run_pre_trade_checks, run_health_checks

BEST_PARAMS_PATH = _PROJECT_ROOT / "results" / "backtesting" / "best_params.json"


def is_rebalance_day(prices) -> bool:
    """
    Es día de rebalanceo si el último día del dataset cambió de mes respecto al anteúltimo.
    Para el primer día del dataset, siempre rebalancear (no hay anteúltimo).
    """
    if len(prices) < 2:
        return True
    last_date = prices.index[-1]
    prev_date = prices.index[-2]
    return (last_date.year, last_date.month) != (prev_date.year, prev_date.month)


def run_paper_trading(force: bool = False):
    print("=== RMT Stat-Arb — Paper Trading ===")

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    if check_data_status(UNIVERSE):
        print("[*] Usando datos locales actualizados a hoy.")
        prices = load_prices()[UNIVERSE]
    else:
        print("[!] Descargando mercado en vivo desde Yahoo Finance …")
        prices = download_prices(UNIVERSE, "2015-01-01")
    prices = prices.dropna(how="all")

    # ── 2. Pre-trade checks ───────────────────────────────────────────────────
    if not run_pre_trade_checks(prices, force=force):
        return

    # ── 3. Parámetros validados ───────────────────────────────────────────────
    if not BEST_PARAMS_PATH.exists():
        print(f"[ERROR] No se encontró {BEST_PARAMS_PATH.name}.")
        print("        Correr primero run_validation_rmt.py para generar best_params.json.")
        return
    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)
    print(f"[*] Parámetros desde {BEST_PARAMS_PATH.name}: {best_params}")

    # ── 4. Detectar día de rebalanceo ─────────────────────────────────────────
    strategy = RMTStrategy(**best_params)
    engine   = PaperEngine(strategy)

    rebalance_day = is_rebalance_day(prices)

    if rebalance_day:
        print("\n[*] Hoy es día de rebalanceo (cambio de mes detectado).")

        # ── Preview de pesos ──────────────────────────────────────────────────
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

        # ── Confirmación humana ───────────────────────────────────────────────
        input("\n[?] ENTER para ejecutar las órdenes en IBKR, CTRL+C para abortar...")

        # ── Ejecución ─────────────────────────────────────────────────────────
        engine.execute(prices)

        # ── Post-trade ────────────────────────────────────────────────────────
        run_health_checks()

    else:
        print("\n[*] Hoy NO es día de rebalanceo. Actualizando NAV sin tocar IBKR…")
        engine.update_no_rebalance(prices)
        print("[*] Para consultar el estado: python -m rmt_stat_arb status")


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser(description="Paper trading RMT Stat-Arb")
    _p.add_argument("--force", action="store_true",
                    help="Saltear check de idempotencia (correr más de una vez por día)")
    _args = _p.parse_args()
    run_paper_trading(force=_args.force)
