"""
PaperEngine para RMT Stat-Arb.

Diferencias respecto al PaperEngine de DQI:
  - execute() lee posiciones de IBKR ANTES de llamar a la estrategia.
  - Convierte quantities → signos (+1/-1) y se los pasa a get_weights().
  - Guard: si get_positions() devuelve {} fuera de bootstrap y el estado
    anterior registra posiciones abiertas, aborta en lugar de continuar.

Lo que es idéntico a DQI:
  - Delta loop (BUY / SELL / HOLD por ticker).
  - _save_daily_log y _save_monitoring_only_log.
  - Manejo de force_liquidate (stop loss).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from engines.api_engine import IBKRClient

# Paths absolutos: este archivo vive en rmt_stat_arb/codigo/engines/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # → rmt_stat_arb/
_RESULTS_PATH = _PROJECT_ROOT / "results" / "paper"

# ─────────────────────────────────────────────────────────────────────────────
# Umbral de cantidad mínima para considerar una posición abierta.
# 0.5 es seguro para acciones enteras (no fraccionarias).
# Si en el futuro se opera con cripto o acciones fraccionarias, bajar a 1e-6.
_MIN_QTY_THRESHOLD = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def _save_monitoring_only_log(estimated_nav, month_start_nav, prices, status):
    """Guarda estado del día sin rebalanceo (días HOLD). Consulta IBKR para actual_weights."""
    _RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    state_path = _RESULTS_PATH / "daily_state.parquet"

    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d %H:%M:%S")
    today_str = now.strftime("%Y-%m-%d")

    # Cargar último target_weights conocido
    last_target_weights = {}
    if state_path.exists():
        existing = pd.read_parquet(state_path)
        if not existing.empty:
            last = existing.iloc[-1]
            try:
                last_target_weights = json.loads(last["target_weights"])
            except (KeyError, TypeError):
                last_target_weights = {}

    # Calcular actual_weights desde IBKR + precios
    actual_weights = {}
    try:
        with IBKRClient() as broker:
            positions = broker.get_positions()
        for ticker, qty in positions.items():
            if abs(qty) <= _MIN_QTY_THRESHOLD or ticker not in prices.columns:
                continue
            price = float(prices[ticker].iloc[-1])
            if price > 0 and estimated_nav > 0:
                actual_weights[ticker] = round((qty * price) / estimated_nav, 6)
    except Exception as e:
        print(f"[Monitoring] No se pudo conectar a IBKR para actual_weights: {e}")

    n_pos = len(actual_weights)

    state = pd.DataFrame([{
        "date":               date_str,
        "estimated_nav":      estimated_nav,
        "month_start_nav":    month_start_nav,
        "sl_triggered":       False,
        "status":             status,
        "target_weights":     json.dumps(last_target_weights),
        "actual_weights":     json.dumps(actual_weights),
        "n_active_positions": n_pos,
    }])
    if state_path.exists():
        existing = pd.read_parquet(state_path)
        existing = existing[~existing["date"].str.startswith(today_str)]
        combined = pd.concat([existing, state], ignore_index=True)
    else:
        combined = state
    combined.to_parquet(state_path)
    combined.to_csv(state_path.with_suffix(".csv"), index=False)


class PaperEngine:
    def __init__(self, strategy):
        self.strategy = strategy
        _RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        prices_hist,
        is_bootstrap:    bool  = False,
        force_liquidate: bool  = False,
        estimated_nav:   float = None,
        month_start_nav: float = None,
        status:          str   = None,
    ):
        """
        Ejecuta un rebalanceo:
          1. Lee posiciones actuales de IBKR.
          2. Convierte quantities → signos y los pasa a get_weights().
          3. Calcula delta y envía órdenes.
          4. Guarda log.

        is_bootstrap: True en el primer run. Permite current_positions={} sin abortar.
        force_liquidate: True si se disparó el stop loss. Cierra todo.
        """
        if estimated_nav is None:
            raise ValueError("estimated_nav es requerido.")
        if month_start_nav is None:
            raise ValueError("month_start_nav es requerido.")

        sl_triggered = force_liquidate

        # ── BLOQUE 1: leer posiciones de IBKR ────────────────────────────────
        print("\n[PaperEngine] Leyendo posiciones actuales de IBKR …")
        with IBKRClient() as broker:
            ibkr_qty = broker.get_positions()   # {ticker: float}

        print(f"[PaperEngine] Posiciones en IBKR: {ibkr_qty}")

        # ── CONVERSIÓN: quantities → signos para get_weights ─────────────────
        #
        # _MIN_QTY_THRESHOLD (= 0.5) filtra ruido de coma flotante.
        # Para acciones enteras, cualquier qty real > 0.5 es una posición real.
        # Ajustar si se opera cripto o acciones fraccionarias.
        current_positions = {
            t: int(np.sign(q))
            for t, q in ibkr_qty.items()
            if abs(q) > _MIN_QTY_THRESHOLD
        }

        # ── GUARD: get_positions vacío fuera de bootstrap ────────────────────
        #
        # Si get_positions() devuelve {} fuera de bootstrap Y el último estado
        # guardado registra posiciones abiertas, es probable que IBKR haya
        # fallado silenciosamente. Operar con current_positions={} en ese caso
        # haría que la estrategia trate todo como primera entrada → duplica exposición.
        if not is_bootstrap and not ibkr_qty:
            last_n_pos = self._last_n_active_positions()
            if last_n_pos > 0:
                print(
                    f"\n[PaperEngine] ERROR: get_positions() devolvió vacío pero "
                    f"el último estado registra {last_n_pos} posición(es) abierta(s).\n"
                    f"    Verificar conexión a TWS. Abortando sin ejecutar órdenes."
                )
                return None

        # ── BLOQUE 2: estrategia + ejecución ─────────────────────────────────
        # La confirmación humana ("¿ejecuto?") vive en run_paper.py, no aquí.
        # execute() es ejecutable sin interacción — compatible con tests y cron.
        if force_liquidate:
            tickers = list(prices_hist.columns)
            target_weights = {t: 0.0 for t in tickers}
            zscores = {}
            print("\n[!] Liquidación: cerrando todas las posiciones.")
        else:
            target_weights, diagnostics = self.strategy.get_weights(
                prices_hist,
                current_positions=current_positions,
                return_diagnostics=True,
            )
            zscores = diagnostics.get("zscores", {})

        execution_log = []
        with IBKRClient() as broker:
            for ticker, weight in target_weights.items():

                # Caso 1: target = 0 → cerrar si hay posición abierta
                if abs(weight) < 1e-6:
                    qty_actual = ibkr_qty.get(ticker, 0.0)
                    if abs(qty_actual) > _MIN_QTY_THRESHOLD:
                        accion     = "SELL" if qty_actual > 0 else "BUY"
                        last_price = float(prices_hist[ticker].iloc[-1])
                        broker.place_order(ticker, accion, abs(int(qty_actual)))
                        execution_log.append({
                            "ticker":   ticker,
                            "action":   accion,
                            "quantity": abs(int(qty_actual)),
                            "target":   0,
                            "current":  int(qty_actual),
                            "weight":   0.0,
                            "price":    last_price,
                            "reason":   "close_position",
                        })
                    continue

                # Caso 2: target ≠ 0 → calcular delta y ejecutar
                live_price = broker.get_price(ticker)
                if pd.isna(live_price) or live_price == 0:
                    live_price = float(prices_hist[ticker].iloc[-1])
                if live_price == 0:
                    continue

                qty_target = int((estimated_nav * weight) / live_price)
                qty_actual = int(ibkr_qty.get(ticker, 0.0))
                delta      = qty_target - qty_actual

                if abs(delta) == 0:
                    execution_log.append({
                        "ticker":   ticker,
                        "action":   "HOLD",
                        "quantity": 0,
                        "target":   qty_target,
                        "current":  qty_actual,
                        "weight":   weight,
                        "price":    live_price,
                        "reason":   "no_change",
                    })
                    continue

                accion = "BUY" if delta > 0 else "SELL"
                broker.place_order(ticker, accion, abs(delta))
                execution_log.append({
                    "ticker":   ticker,
                    "action":   accion,
                    "quantity": abs(delta),
                    "target":   qty_target,
                    "current":  qty_actual,
                    "weight":   weight,
                    "price":    live_price,
                    "reason":   "rebalance",
                })

        self._save_daily_log(
            target_weights, execution_log, estimated_nav,
            month_start_nav, sl_triggered, status, zscores,
        )
        return target_weights

    # ── Preview (sin ejecución) ───────────────────────────────────────────────

    def compute_target_weights(self, prices_hist) -> tuple[dict, dict]:
        """
        Calcula los pesos objetivo SIN ejecutar órdenes.

        Usado por run_paper.py para mostrar el preview al operador antes de
        confirmar. No requiere interacción humana — testeable de forma aislada.

        Devuelve (target_weights, current_positions) donde:
          target_weights    : {ticker: float}  con signo y sizing
          current_positions : {ticker: ±1}     posiciones actuales en IBKR
        """
        with IBKRClient() as broker:
            ibkr_qty = broker.get_positions()

        current_positions = {
            t: int(np.sign(q))
            for t, q in ibkr_qty.items()
            if abs(q) > _MIN_QTY_THRESHOLD
        }

        target_weights = self.strategy.get_weights(
            prices_hist,
            current_positions=current_positions,
        )
        return target_weights, current_positions

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _last_n_active_positions(self) -> int:
        """Lee el último estado guardado y devuelve n_active_positions."""
        state_path = _RESULTS_PATH / "daily_state.parquet"
        if not state_path.exists():
            return 0
        try:
            df = pd.read_parquet(state_path)
            if df.empty:
                return 0
            return int(df.iloc[-1]["n_active_positions"])
        except Exception:
            return 0

    def _save_daily_log(
        self, target_weights, execution_log, estimated_nav,
        month_start_nav, sl_triggered, status, zscores=None,
    ):
        now       = datetime.now()
        date_str  = now.strftime("%Y-%m-%d %H:%M:%S")
        today_str = now.strftime("%Y-%m-%d")

        clean_weights = {k: float(v) for k, v in target_weights.items()}

        # daily_state.parquet (siempre)
        state_path = _RESULTS_PATH / "daily_state.parquet"
        state = pd.DataFrame([{
            "date":               date_str,
            "estimated_nav":      estimated_nav,
            "month_start_nav":    month_start_nav,
            "sl_triggered":       sl_triggered,
            "status":             status,
            "target_weights":     json.dumps(clean_weights),
            "actual_weights":     json.dumps(clean_weights),
            "n_active_positions": sum(1 for w in clean_weights.values() if abs(w) > 1e-6),
            "zscores":            json.dumps(zscores or {}),
        }])
        if state_path.exists():
            existing = pd.read_parquet(state_path)
            existing = existing[~existing["date"].str.startswith(today_str)]
            combined = pd.concat([existing, state], ignore_index=True)
        else:
            combined = state
        combined.to_parquet(state_path)
        combined.to_csv(state_path.with_suffix(".csv"), index=False)

        # orders_log + execution_log.jsonl (solo si hubo órdenes reales)
        real_orders = [e for e in execution_log if e.get("action") in ("BUY", "SELL")]
        if real_orders:
            orders_path = _RESULTS_PATH / "orders_log.parquet"
            new_orders      = pd.DataFrame(real_orders)
            new_orders["date"] = date_str
            if orders_path.exists():
                existing = pd.read_parquet(orders_path)
                existing = existing[~existing["date"].str.startswith(today_str)]
                combined = pd.concat([existing, new_orders], ignore_index=True)
            else:
                combined = new_orders
            combined.to_parquet(orders_path)
            combined.to_csv(orders_path.with_suffix(".csv"), index=False)

            log_path = _RESULTS_PATH / "execution_log.jsonl"
            lines = []
            if log_path.exists():
                with open(log_path, "r") as f:
                    lines = [
                        line for line in f
                        if not json.loads(line)["date"].startswith(today_str)
                    ]
            with open(log_path, "w") as f:
                for line in lines:
                    f.write(line)
                f.write(json.dumps({
                    "date":            date_str,
                    "estimated_nav":   estimated_nav,
                    "month_start_nav": month_start_nav,
                    "sl_triggered":    sl_triggered,
                    "status":          status,
                    "target_weights":  clean_weights,
                    "actual_weights":  clean_weights,
                    "execution":       real_orders,
                }) + "\n")

        print(f"[PaperEngine] Logs guardados en {_RESULTS_PATH}")