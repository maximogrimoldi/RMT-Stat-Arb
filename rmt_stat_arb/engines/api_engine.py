"""
IBKRClient: conexión a TWS via ibapi para obtener precios y enviar órdenes.
Solo paper trading — no cambies port=7497 por 7496 sin quererlo.
"""

import threading
import time
import math

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order


# ── Wrappers internos ─────────────────────────────────────────────────────────

class _SuppressInfoErrors(EWrapper):
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if 2100 <= errorCode < 2200:
            return
        print(f"[IBKR] Error {errorCode}: {errorString}")


class _PrecioWrapper(EClient, _SuppressInfoErrors):
    """Recibe un bar histórico y guarda el close. Un solo uso — conectar, pedir, desconectar."""
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = False
        self.close = math.nan
        self._done = False

    def nextValidId(self, orderId): self.ready = True
    def historicalData(self, reqId, bar): self.close = float(bar.close)
    def historicalDataEnd(self, reqId, start, end): self._done = True


class _OrdenWrapper(EClient, _SuppressInfoErrors):
    """Envía una orden de mercado y captura el nextValidId para el orderId."""
    def __init__(self):
        EClient.__init__(self, self)
        self.ready    = False
        self.order_id = None

    def nextValidId(self, orderId):
        self.order_id = orderId
        self.ready    = True


class _PositionsWrapper(EClient, _SuppressInfoErrors):
    """Recibe el snapshot de posiciones desde TWS."""
    def __init__(self):
        EClient.__init__(self, self)
        self.ready = False
        self.positions = {}
        self._done = False

    def nextValidId(self, orderId):
        self.ready = True

    def position(self, account, contract, position, avgCost):
        self.positions[contract.symbol] = float(position)

    def positionEnd(self):
        self._done = True


class _AccountValueWrapper(EClient, _SuppressInfoErrors):
    """Recibe el resumen de cuenta (NetLiquidation) desde TWS."""
    def __init__(self):
        EClient.__init__(self, self)
        self.ready  = False
        self.nav    = math.nan
        self._done  = False

    def nextValidId(self, orderId):
        self.ready = True

    def accountSummary(self, reqId, account, tag, value, currency):
        if tag == "NetLiquidation":
            try:
                self.nav = float(value)
            except ValueError:
                pass

    def accountSummaryEnd(self, reqId):
        self._done = True


# ── Cliente principal ─────────────────────────────────────────────────────────

class IBKRClient:
    """
    Cliente ligero para TWS. Soporta context manager (with IBKRClient() as ib:).
    Internamente crea una conexión fresca por operación — mismo patrón que el archivo de referencia.
    """

    def __init__(self, host="127.0.0.1", port=7497, client_id=55, timeout=12.0):
        self.host       = host
        self.port       = port
        self.client_id  = client_id
        self.timeout    = timeout
        self._connected = False

    def connect(self):
        """Conecta a TWS. Lanza ConnectionError si no responde en timeout."""
        probe = _PrecioWrapper()
        probe.connect(self.host, self.port, self.client_id)
        threading.Thread(target=probe.run, daemon=True).start()
        t0 = time.time()
        while not probe.ready and time.time() - t0 < self.timeout:
            time.sleep(0.05)
        probe.disconnect()
        if not probe.ready:
            raise ConnectionError("TWS no está disponible. Abrí Trader Workstation antes de correr el sistema.")
        self._connected = True
        print(f"[IBKRClient] Conectado a TWS ({self.host}:{self.port})")

    def disconnect(self):
        """Marca el cliente como desconectado (las conexiones internas ya se cierran solas)."""
        self._connected = False
        print("[IBKRClient] Desconectado.")

    def get_price(self, ticker):
        """Devuelve el último precio de cierre diario de un ticker via TWS."""
        if not self._connected:
            return math.nan
        try:
            app = _PrecioWrapper()
            # client_id dinámico para evitar colisiones cuando TWS tarda en liberar el id anterior
            app.connect(self.host, self.port, self.client_id + 100 + int(time.time() % 1000))
            threading.Thread(target=app.run, daemon=True).start()

            t0 = time.time()
            while not app.ready and time.time() - t0 < self.timeout:
                time.sleep(0.05)

            c = Contract()
            c.symbol, c.secType, c.currency, c.exchange = ticker.upper(), "STK", "USD", "SMART"

            app.reqHistoricalData(
                reqId=1, contract=c, endDateTime="",
                durationStr="1 D", barSizeSetting="1 day",
                whatToShow="TRADES", useRTH=1, formatDate=2,
                keepUpToDate=False, chartOptions=[],
            )

            t0 = time.time()
            while not app._done and time.time() - t0 < self.timeout:
                time.sleep(0.05)

            px = app.close
            app.disconnect()
            return px
        except Exception as e:
            print(f"[IBKRClient] get_price({ticker}) falló: {e}")
            return math.nan

    def place_order(self, ticker, lado, cantidad):
        """Envía una orden de mercado a TWS. lado = 'BUY' o 'SELL'. Solo paper trading."""
        if not self._connected:
            print(f"[IBKRClient] Sin conexión — orden {lado} {cantidad}x{ticker} no enviada.")
            return
        try:
            app = _OrdenWrapper()
            # Le sumamos un random al client_id para que TWS no se confunda al reconectar tan rápido
            app.connect(self.host, self.port, self.client_id + 200 + int(time.time() % 1000))
            threading.Thread(target=app.run, daemon=True).start()

            t0 = time.time()
            while not app.ready and time.time() - t0 < self.timeout:
                time.sleep(0.05)

            if not app.ready:
                print(f"[IBKRClient] place_order: timeout esperando nextValidId para {ticker}.")
                app.disconnect()
                return

            c = Contract()
            c.symbol, c.secType, c.currency, c.exchange = ticker.upper(), "STK", "USD", "SMART"

            o = Order()
            o.action        = lado.upper()  # "BUY" o "SELL"
            o.orderType     = "MKT"
            o.totalQuantity = cantidad
            o.tif           = "DAY"
            
            o.eTradeOnly = False
            o.firmQuoteOnly = False

            app.placeOrder(app.order_id, c, o)
            time.sleep(1)  # 1 segundo es suficiente, tenemos que apurarnos
            print(f"[IBKRClient] Orden enviada: {lado.upper()} {cantidad}x {ticker}")
            app.disconnect()
        except Exception as e:
            print(f"[IBKRClient] place_order({ticker}) falló: {e}")
    
    def get_account_value(self) -> float:
        """Pide NetLiquidation a IBKR. Retorna el NAV total de la cuenta en USD."""
        if not self._connected:
            print("[IBKRClient] Sin conexión — get_account_value devuelve nan.")
            return math.nan
        try:
            app = _AccountValueWrapper()
            app.connect(self.host, self.port, self.client_id + 400 + int(time.time() % 1000))
            threading.Thread(target=app.run, daemon=True).start()

            t0 = time.time()
            while not app.ready and time.time() - t0 < self.timeout:
                time.sleep(0.05)

            if not app.ready:
                print("[IBKRClient] get_account_value: timeout esperando nextValidId.")
                app.disconnect()
                return math.nan

            app.reqAccountSummary(reqId=1, groupName="All", tags="NetLiquidation")

            t0 = time.time()
            while not app._done and time.time() - t0 < self.timeout:
                time.sleep(0.05)

            nav = app.nav
            app.reqAccountSummaryCancel(reqId=1)
            app.disconnect()
            return nav
        except Exception as e:
            print(f"[IBKRClient] get_account_value falló: {e}")
            return math.nan

    def get_positions(self):
        """Devuelve dict {ticker: cantidad} de las posiciones actuales en TWS paper."""
        if not self._connected:
            print("[IBKRClient] Sin conexión — get_positions devuelve dict vacío.")
            return {}
        try:
            app = _PositionsWrapper()
            # client_id dinámico para evitar colisiones cuando TWS tarda en liberar el id anterior
            app.connect(self.host, self.port, self.client_id + 300 + int(time.time() % 1000))
            threading.Thread(target=app.run, daemon=True).start()

            t0 = time.time()
            while not app.ready and time.time() - t0 < self.timeout:
                time.sleep(0.05)

            if not app.ready:
                print("[IBKRClient] get_positions: timeout esperando nextValidId.")
                app.disconnect()
                return {}

            app.reqPositions()

            t0 = time.time()
            while not app._done and time.time() - t0 < self.timeout:
                time.sleep(0.05)

            positions = dict(app.positions)
            app.disconnect()
            return positions
        except Exception as e:
            print(f"[IBKRClient] get_positions falló: {e}")
            return {}

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
