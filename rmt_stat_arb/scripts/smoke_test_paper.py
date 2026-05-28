"""
Smoke test del flujo de paper trading RMT — sin TWS real.

Mockeá IBKRClient en engines.paper_engine para que nunca toque el socket.
Usa los precios reales ya descargados en data/storage/prices.parquet.

Escenarios:
  (a) HOLD:       día sin rebalanceo → monitoring log + health checks, cero órdenes
  (b) Bootstrap:  is_bootstrap=True, posiciones vacías → pesos RMT reales + execute
  (c) Guard:      is_bootstrap=False, posiciones vacías, estado previo > 0 → abortar

Cada escenario atrapa sus propias excepciones; el script siempre llega al resumen.
"""

import sys, json, shutil, traceback
from pathlib import Path
from unittest.mock import patch
import pandas as pd

# ── sys.path ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # → rmt_stat_arb/
_CODIGO_DIR   = _PROJECT_ROOT / "codigo"
for _p in [str(_CODIGO_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.ingest          import load_prices
from data.universe        import UNIVERSE
from strategy.core        import RMTStrategy
from engines.paper_engine import PaperEngine, _save_monitoring_only_log
from monitoring.checks    import run_health_checks

# ── Datos reales desde disco ──────────────────────────────────────────────────
print("[Setup] Cargando precios desde disco…")
prices = load_prices()[UNIVERSE]
print(f"[Setup] {prices.shape[1]} tickers × {len(prices)} días "
      f"({prices.index[0].date()} → {prices.index[-1].date()})")

# ── Directorio temporal aislado para artefactos del smoke test ────────────────
_TEST_DIR = _PROJECT_ROOT / "results" / "_smoke_paper"
_TEST_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_NAV = 100_000.0

# ── Mock IBKRClient ───────────────────────────────────────────────────────────
# Reemplaza la clase completa en el namespace de engines.paper_engine.
# Cada instancia recibe un dict de posiciones y una lista donde acumula órdenes.

def make_mock_ibkr(positions: dict, orders: list):
    """Devuelve una clase compatible con IBKRClient pero sin sockets."""
    _prices = prices

    class _MockIBKRClient:
        def __enter__(self):  return self
        def __exit__(self, *a): return False
        def get_positions(self): return dict(positions)
        def get_price(self, ticker):
            return float(_prices[ticker].iloc[-1]) if ticker in _prices.columns else 100.0
        def place_order(self, ticker, action, qty):
            orders.append({"ticker": ticker, "action": action, "qty": qty})
            print(f"    [MOCK ORDER] {action:4s}  {qty:5d}  {ticker}")

    return _MockIBKRClient


# ── Helpers de escenario ──────────────────────────────────────────────────────

def _clean():
    for f in _TEST_DIR.iterdir():
        f.unlink()

def _write_prev_state(n_pos: int):
    """Escribe un daily_state.parquet simulando n posiciones abiertas ayer."""
    prev = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    weights = {"AAPL": 0.1, "MSFT": -0.1, "GOOGL": 0.1,
               "JPM": -0.1, "BAC": 0.1} if n_pos >= 5 else {}
    pd.DataFrame([{
        "date":               prev,
        "estimated_nav":      INITIAL_NAV,
        "month_start_nav":    INITIAL_NAV,
        "sl_triggered":       False,
        "status":             "REBALANCE - Bootstrap",
        "target_weights":     json.dumps(weights),
        "actual_weights":     json.dumps(weights),
        "n_active_positions": n_pos,
        "zscores":            json.dumps({"AAPL": 2.1, "MSFT": -1.9}),
    }]).to_parquet(_TEST_DIR / "daily_state.parquet")


RESULTS: dict[str, str] = {}


# ══════════════════════════════════════════════════════════════════════════════
# (a) HOLD — día sin rebalanceo
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  ESCENARIO (a): HOLD — día sin rebalanceo")
print("═"*60)
try:
    _clean()
    orders_a: list = []
    MockIBKR_a = make_mock_ibkr(positions={}, orders=orders_a)

    with patch("engines.paper_engine.IBKRClient",       MockIBKR_a), \
         patch("engines.paper_engine._RESULTS_PATH",    _TEST_DIR), \
         patch("monitoring.checks._STATE_PATH",         _TEST_DIR / "daily_state.parquet"), \
         patch("monitoring.checks._ORDERS_PATH",        _TEST_DIR / "orders_log.parquet"):

        _save_monitoring_only_log(INITIAL_NAV, INITIAL_NAV, prices,
                                  "HOLD - Día normal (Smoke)")

        state_path = _TEST_DIR / "daily_state.parquet"
        assert state_path.exists(), "daily_state.parquet no fue creado"
        df = pd.read_parquet(state_path)
        assert not df.empty
        print(f"  ✓ monitoring log escrito — {len(df)} fila(s), "
              f"columnas: {list(df.columns)}")

        assert len(orders_a) == 0, f"Órdenes inesperadas: {orders_a}"
        print("  ✓ cero órdenes enviadas")

        run_health_checks()
        print("  ✓ health checks corrieron sin excepción")

    RESULTS["a"] = "PASS"
except Exception:
    print("  ✗ EXCEPCIÓN:")
    traceback.print_exc()
    RESULTS["a"] = "FAIL"


# ══════════════════════════════════════════════════════════════════════════════
# (b) BOOTSTRAP — posiciones vacías, calcular pesos RMT reales
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  ESCENARIO (b): BOOTSTRAP — posiciones vacías, pesos RMT reales")
print("═"*60)
try:
    _clean()
    orders_b: list = []
    MockIBKR_b = make_mock_ibkr(positions={}, orders=orders_b)

    strategy = RMTStrategy(entry_threshold=2.0, exit_threshold=1.0,
                           ventana_betas=252, ventana_zscore=252,
                           sizing_by_zscore=False)

    print("  [*] Calculando pesos RMT sobre datos reales (puede tardar ~30 s)…")

    with patch("engines.paper_engine.IBKRClient",       MockIBKR_b), \
         patch("engines.paper_engine._RESULTS_PATH",    _TEST_DIR), \
         patch("monitoring.checks._STATE_PATH",         _TEST_DIR / "daily_state.parquet"), \
         patch("monitoring.checks._ORDERS_PATH",        _TEST_DIR / "orders_log.parquet"), \
         patch("builtins.input", return_value=""):

        engine = PaperEngine(strategy)
        result = engine.execute(
            prices,
            is_bootstrap    = True,
            force_liquidate = False,
            estimated_nav   = INITIAL_NAV,
            month_start_nav = INITIAL_NAV,
            status          = "REBALANCE - Bootstrap (Smoke)",
        )

        assert result is not None, "execute() devolvió None — guard disparado inesperadamente"

        longs  = {t: w for t, w in result.items() if w >  1e-6}
        shorts = {t: w for t, w in result.items() if w < -1e-6}
        print(f"  ✓ pesos calculados — {len(longs)} longs, {len(shorts)} shorts")
        print(f"    net={sum(result.values()):+.4f}  "
              f"gross={sum(abs(w) for w in result.values()):.4f}")

        buys  = [o for o in orders_b if o["action"] == "BUY"]
        sells = [o for o in orders_b if o["action"] == "SELL"]
        print(f"  ✓ órdenes mock: {len(buys)} BUY, {len(sells)} SELL")

        state_path = _TEST_DIR / "daily_state.parquet"
        assert state_path.exists()
        df = pd.read_parquet(state_path)
        assert "zscores" in df.columns, "columna zscores ausente en el parquet"
        zs = json.loads(df.iloc[-1]["zscores"])
        print(f"  ✓ daily_state.parquet escrito — {df.shape[1]} columnas, "
              f"{len(zs)} z-scores registrados")

        run_health_checks()
        print("  ✓ health checks corrieron sin excepción")

    RESULTS["b"] = "PASS"
except Exception:
    print("  ✗ EXCEPCIÓN:")
    traceback.print_exc()
    RESULTS["b"] = "FAIL"


# ══════════════════════════════════════════════════════════════════════════════
# (c) GUARD — get_positions={} fuera de bootstrap con estado previo > 0
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  ESCENARIO (c): GUARD — posiciones vacías, estado previo registra 5 abiertas")
print("═"*60)
try:
    _clean()
    _write_prev_state(n_pos=5)   # simular 5 posiciones abiertas ayer

    orders_c: list = []
    MockIBKR_c = make_mock_ibkr(positions={}, orders=orders_c)

    strategy = RMTStrategy(entry_threshold=2.0, exit_threshold=1.0)

    with patch("engines.paper_engine.IBKRClient",    MockIBKR_c), \
         patch("engines.paper_engine._RESULTS_PATH", _TEST_DIR):

        engine = PaperEngine(strategy)
        result = engine.execute(
            prices,
            is_bootstrap    = False,      # ← guard activo
            force_liquidate = False,
            estimated_nav   = INITIAL_NAV,
            month_start_nav = INITIAL_NAV,
            status          = "REBALANCE - Smoke Guard",
        )

    assert result is None, \
        f"Se esperaba None del guard, execute() devolvió {type(result)}"
    assert len(orders_c) == 0, \
        f"El guard debió abortar antes de enviar órdenes: {orders_c}"
    print("  ✓ execute() devolvió None — guard activado correctamente")
    print("  ✓ cero órdenes enviadas")

    RESULTS["c"] = "PASS"
except Exception:
    print("  ✗ EXCEPCIÓN:")
    traceback.print_exc()
    RESULTS["c"] = "FAIL"


# ── Cleanup ───────────────────────────────────────────────────────────────────
shutil.rmtree(_TEST_DIR, ignore_errors=True)

# ── Resumen ───────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  RESUMEN SMOKE TEST")
print("═"*60)
labels = [("a", "HOLD"), ("b", "Bootstrap"), ("c", "Guard")]
for key, name in labels:
    status = RESULTS.get(key, "NO EJECUTADO")
    icon   = "✓" if status == "PASS" else "✗"
    print(f"  {icon}  ({key}) {name:<14}  {status}")
print("═"*60)

all_pass = all(RESULTS.get(k) == "PASS" for k, _ in labels)
sys.exit(0 if all_pass else 1)
