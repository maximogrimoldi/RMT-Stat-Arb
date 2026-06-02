"""
PaperEngine para RMT Stat-Arb.

NAV se trackea internamente con marking-to-market:
  NAV_hoy = NAV_ayer × (1 + Σ actual_weights_ayer[t] × ret_diario[t])
Esto aísla el P&L de la sub-estrategia RMT de cualquier otra posición en la cuenta.

execute():
  1. Estima NAV con marking-to-market (ANTES de ejecutar órdenes).
  2. Lee posiciones de IBKR.
  3. Guard de posiciones vacías inesperadas.
  4. Calcula target_weights via strategy.get_weights().
  5. Ejecuta el delta de órdenes.
  6. Persiste en results/trading/daily_state.parquet.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from constants import INITIAL_CAPITAL
from engines.api_engine import IBKRClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # → rmt_stat_arb/
_RESULTS_PATH = _PROJECT_ROOT / "results" / "trading"
_STATE_PATH   = _RESULTS_PATH / "daily_state.parquet"

_MIN_QTY_THRESHOLD = 0.5


class PaperEngine:
    INITIAL_CAPITAL = INITIAL_CAPITAL   # fuente única: rmt_stat_arb/constants.py

    def __init__(self, strategy):
        self.strategy = strategy
        _RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # ── NAV con marking-to-market ─────────────────────────────────────────────

    def _estimate_nav(self, prices_hist) -> tuple[float, float]:
        """
        Primer run: NAV = INITIAL_CAPITAL.
        Runs siguientes: NAV_hoy = NAV_ayer × (1 + ret_portafolio_diario).
          ret_portafolio = Σ (actual_weights_ayer[t] × ret_diario[t])
          Funciona con pesos negativos (shorts) por el dot-product con signo.
        Retorna (estimated_nav, month_start_nav).
        """
        if not _STATE_PATH.exists():
            return self.INITIAL_CAPITAL, self.INITIAL_CAPITAL

        df = pd.read_parquet(_STATE_PATH)
        if df.empty:
            return self.INITIAL_CAPITAL, self.INITIAL_CAPITAL

        last        = df.iloc[-1]
        nav_ayer    = float(last["estimated_nav"])
        weights_ayer = json.loads(last["actual_weights"])

        daily_rets = prices_hist.pct_change().iloc[-1]
        ret_portafolio = sum(
            weights_ayer.get(t, 0.0) * float(daily_rets[t])
            for t in weights_ayer
            if t in daily_rets.index and not pd.isna(daily_rets[t])
        )
        estimated_nav = nav_ayer * (1 + ret_portafolio)

        today_month   = datetime.now().strftime("%Y-%m")
        df_this_month = df[df["date_only"].str.startswith(today_month)]
        if df_this_month.empty:
            month_start_nav = estimated_nav   # primer run del mes
        else:
            month_start_nav = float(df_this_month.iloc[0]["month_start_nav"])

        return estimated_nav, month_start_nav

    # ── Execute ───────────────────────────────────────────────────────────────

    def execute(self, prices_hist) -> dict | None:
        """
        Rebalanceo completo. Retorna target_weights o None si se abortó.
        La confirmación humana vive en run_paper.py — este método es no-interactivo.
        """
        # ── 1. NAV con marking-to-market ──────────────────────────────────────
        estimated_nav, month_start_nav = self._estimate_nav(prices_hist)
        print(f"\n[PaperEngine] NAV (marking-to-market) : ${estimated_nav:,.2f}")
        print(f"[PaperEngine] Month start NAV          : ${month_start_nav:,.2f}")

        # ── 2. Posiciones IBKR ────────────────────────────────────────────────
        print("[PaperEngine] Leyendo posiciones de IBKR …")
        with IBKRClient() as broker:
            ibkr_qty = broker.get_positions()
        print(f"[PaperEngine] Posiciones: {ibkr_qty}")

        current_positions = {
            t: int(np.sign(q))
            for t, q in ibkr_qty.items()
            if abs(q) > _MIN_QTY_THRESHOLD
        }

        # ── 3. Guard: posiciones vacías inesperadas ───────────────────────────
        last_n_pos = self._last_n_active_positions()
        if not ibkr_qty and last_n_pos > 0:
            print(
                f"\n[PaperEngine] ERROR: get_positions() vacío pero el último estado "
                f"registra {last_n_pos} posición(es). Verificar conexión a TWS. Abortando."
            )
            return None

        # ── 4. Estrategia ─────────────────────────────────────────────────────
        target_weights, diagnostics = self.strategy.get_weights(
            prices_hist,
            current_positions=current_positions,
            return_diagnostics=True,
        )
        zscores = diagnostics.get("zscores", {})

        # ── 5. Ejecutar delta de órdenes ──────────────────────────────────────
        execution_log = []
        with IBKRClient() as broker:
            for ticker, weight in target_weights.items():

                # Cerrar posición
                if abs(weight) < 1e-6:
                    qty_actual = ibkr_qty.get(ticker, 0.0)
                    if abs(qty_actual) > _MIN_QTY_THRESHOLD:
                        accion     = "SELL" if qty_actual > 0 else "BUY"
                        last_price = float(prices_hist[ticker].iloc[-1])
                        broker.place_order(ticker, accion, abs(int(qty_actual)))
                        execution_log.append({
                            "ticker": ticker, "action": accion,
                            "quantity": abs(int(qty_actual)), "target": 0,
                            "current": int(qty_actual), "weight": 0.0,
                            "price": last_price, "reason": "close_position",
                        })
                    continue

                # Ajustar posición
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
                        "ticker": ticker, "action": "HOLD", "quantity": 0,
                        "target": qty_target, "current": qty_actual,
                        "weight": weight, "price": live_price, "reason": "no_change",
                    })
                    continue

                accion = "BUY" if delta > 0 else "SELL"
                broker.place_order(ticker, accion, abs(delta))
                execution_log.append({
                    "ticker": ticker, "action": accion, "quantity": abs(delta),
                    "target": qty_target, "current": qty_actual,
                    "weight": weight, "price": live_price, "reason": "rebalance",
                })

        # ── 6. Persistir ──────────────────────────────────────────────────────
        self._save_daily_state(
            estimated_nav   = estimated_nav,
            month_start_nav = month_start_nav,
            target_weights  = target_weights,
            actual_weights  = target_weights,   # rebalanceo perfecto
            zscores         = zscores,
        )
        self._save_orders_log(execution_log)
        return target_weights

    # ── Preview (sin ejecución) ───────────────────────────────────────────────

    def compute_target_weights(self, prices_hist) -> tuple[dict, dict]:
        """
        Calcula pesos SIN ejecutar órdenes ni guardar estado. Para el preview.
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

    # ── Update sin rebalanceo ─────────────────────────────────────────────────

    def update_no_rebalance(self, prices_hist) -> dict | None:
        """
        Actualiza NAV con marking-to-market sin tocar IBKR.
        Hereda target_weights del último rebalanceo, recalcula actual_weights por drift,
        calcula zscores frescos. Appendea fila al daily_state.
        Retorna dict con resumen o None si falla.
        """
        if not _STATE_PATH.exists():
            print("[PaperEngine] ERROR: no hay estado previo. Correr un rebalanceo primero.")
            return None

        df = pd.read_parquet(_STATE_PATH)
        if df.empty:
            print("[PaperEngine] ERROR: daily_state vacío. Correr un rebalanceo primero.")
            return None

        last = df.iloc[-1]

        # NAV con marking-to-market (reusa el helper existente)
        estimated_nav, month_start_nav = self._estimate_nav(prices_hist)
        nav_ayer = float(last["estimated_nav"])
        daily_return = (estimated_nav / nav_ayer) - 1.0

        # Heredar target_weights del último rebalanceo
        last_rebal_target_weights = self._get_last_rebalance_target_weights(df)

        # Calcular actual_weights por drift
        weights_ayer = json.loads(last["actual_weights"])
        daily_rets   = prices_hist.pct_change().iloc[-1]
        actual_weights = {}
        for ticker, w_ayer in weights_ayer.items():
            ret_t = float(daily_rets[ticker]) if ticker in daily_rets.index and not pd.isna(daily_rets[ticker]) else 0.0
            denom = 1.0 + daily_return if (1.0 + daily_return) != 0 else 1.0
            actual_weights[ticker] = w_ayer * (1 + ret_t) / denom

        # Z-scores frescos: llamar a la estrategia solo para diagnósticos
        current_positions = {
            t: int(np.sign(w))
            for t, w in actual_weights.items()
            if abs(w) > 1e-6
        }
        _, diagnostics = self.strategy.get_weights(
            prices_hist,
            current_positions=current_positions,
            return_diagnostics=True,
        )
        zscores = diagnostics.get("zscores", {})

        # Persistir
        self._save_daily_state(
            estimated_nav   = estimated_nav,
            month_start_nav = month_start_nav,
            target_weights  = last_rebal_target_weights,
            actual_weights  = actual_weights,
            zscores         = zscores,
        )

        # Imprimir resumen
        print("\n" + "═" * 51)
        print("  STATUS UPDATE — no es día de rebalanceo")
        print("═" * 51)
        print(f"  NAV ayer       : ${nav_ayer:,.2f}")
        print(f"  NAV hoy        : ${estimated_nav:,.2f}")
        print(f"  Retorno diario : {daily_return:+.4%}")
        print(f"  Drawdown mes   : {(estimated_nav/month_start_nav - 1):+.2%}")
        print("═" * 51 + "\n")

        return {
            "nav_hoy":        estimated_nav,
            "nav_ayer":       nav_ayer,
            "daily_return":   daily_return,
            "target_weights": last_rebal_target_weights,
            "actual_weights": actual_weights,
        }

    def _get_last_rebalance_target_weights(self, df) -> dict:
        """
        Devuelve los target_weights del primer rebalanceo del mes actual.
        Si no hay filas del mes actual, devuelve los de la última fila.
        """
        last_date     = df.iloc[-1]["date_only"]
        current_month = last_date[:7]   # 'YYYY-MM'
        month_rows    = df[df["date_only"].str.startswith(current_month)]
        if month_rows.empty:
            return json.loads(df.iloc[-1]["target_weights"])
        return json.loads(month_rows.iloc[0]["target_weights"])

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _save_daily_state(
        self,
        estimated_nav:   float,
        month_start_nav: float,
        target_weights:  dict,
        actual_weights:  dict,
        zscores:         dict,
    ) -> None:
        """
        Agrega una fila al daily_state.parquet (append-only).
        Exporta también CSV para inspección visual.
        """
        now      = datetime.now()
        date_str = now.strftime("%Y-%m-%d %H:%M:%S")

        clean_tw = {k: float(v) for k, v in target_weights.items()}
        clean_aw = {k: float(v) for k, v in actual_weights.items()}
        n_active = sum(1 for w in clean_tw.values() if abs(w) > 1e-6)

        row = pd.DataFrame([{
            "date":               date_str,
            "date_only":          date_str[:10],
            "estimated_nav":      estimated_nav,
            "month_start_nav":    month_start_nav,
            "target_weights":     json.dumps(clean_tw),
            "actual_weights":     json.dumps(clean_aw),
            "n_active_positions": n_active,
            "zscores":            json.dumps(zscores or {}),
        }])

        if _STATE_PATH.exists():
            existing = pd.read_parquet(_STATE_PATH)
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            combined = row

        combined.to_parquet(_STATE_PATH)
        combined.to_csv(_STATE_PATH.with_suffix(".csv"), index=False)
        print(f"[PaperEngine] Estado guardado en {_STATE_PATH}  (fila {len(combined)})")

    def _last_n_active_positions(self) -> int:
        """Devuelve n_active_positions de la última fila del parquet (0 si no existe)."""
        if not _STATE_PATH.exists():
            return 0
        try:
            df = pd.read_parquet(_STATE_PATH)
            if df.empty:
                return 0
            return int(df.iloc[-1]["n_active_positions"])
        except Exception:
            return 0

    def _save_orders_log(self, execution_log: list) -> None:
        real_orders = [e for e in execution_log if e.get("action") in ("BUY", "SELL")]
        if not real_orders:
            return

        date_str    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        today_str   = date_str[:10]
        orders_path = _RESULTS_PATH / "orders_log.parquet"
        new_orders  = pd.DataFrame(real_orders)
        new_orders["date"] = date_str

        if orders_path.exists():
            existing = pd.read_parquet(orders_path)
            existing = existing[~existing["date"].str.startswith(today_str)]
            combined = pd.concat([existing, new_orders], ignore_index=True)
        else:
            combined = new_orders

        combined.to_parquet(orders_path)
        combined.to_csv(orders_path.with_suffix(".csv"), index=False)
