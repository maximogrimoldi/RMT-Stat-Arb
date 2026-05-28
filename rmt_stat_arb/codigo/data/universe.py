"""
Universo de acciones para RMT Stat-Arb.

Criterios de selección:
  - ~100 acciones del S&P 500, balanceadas por sector GICS.
  - Liquidez alta: todas son large/mid cap con volumen diario suficiente para
    stat-arb (bid-ask spreads bajos, sin riesgo de market impact).
  - Historia mínima: cotizan desde antes de 2015 (suficiente para ventana de
    252 días más margen de arranque).
  - Homogeneidad deliberada: acciones del S&P 500, NO ETFs heterogéneos.
    RMT necesita activos que compartan factores comunes (mercado, sector, macro)
    para que Marchenko-Pastur separe señal estructural de ruido idiosincrático.
    Mezclar acciones con ETFs de renta fija o commodities rompería esa supuesto.

Criterio de exclusión:
  - NO se seleccionaron por rendimiento pasado (eso sería look-ahead bias).
  - Se excluyeron tickers con IPO posterior a 2015 (ej: UBER 2019, ABNB 2020).
  - Se excluyeron tickers con cambios de ticker posterior a 2015 que yfinance
    no resuelve históricamente (ej: LIN post-merger 2018 reemplaza a Praxair).
"""

# ── Technology / Software ─────────────────────────────────────────────────────
TECH = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "META",   # Meta Platforms  (IPO 2012)
    "ORCL",   # Oracle
    "CRM",    # Salesforce
    "ADBE",   # Adobe
    "IBM",    # IBM
    "INTC",   # Intel
    "CSCO",   # Cisco
]

# ── Semiconductors ────────────────────────────────────────────────────────────
SEMIS = [
    "NVDA",   # NVIDIA
    "AMD",    # Advanced Micro Devices
    "AVGO",   # Broadcom  (Avago IPO 2009, ticker AVGO continuo)
    "QCOM",   # Qualcomm
    "TXN",    # Texas Instruments
    "AMAT",   # Applied Materials
    "KLAC",   # KLA Corporation
    "LRCX",   # Lam Research
    "MU",     # Micron Technology
]

# ── Financials ────────────────────────────────────────────────────────────────
FINANCIALS = [
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "C",      # Citigroup
    "BLK",    # BlackRock
    "SCHW",   # Charles Schwab
    "AXP",    # American Express
    "SPGI",   # S&P Global
    "MCO",    # Moody's
    "ICE",    # Intercontinental Exchange  (IPO 2013)
    "CME",    # CME Group
    "USB",    # U.S. Bancorp
    "PNC",    # PNC Financial
]

# ── Healthcare ────────────────────────────────────────────────────────────────
HEALTHCARE = [
    "UNH",    # UnitedHealth
    "JNJ",    # Johnson & Johnson
    "LLY",    # Eli Lilly
    "ABBV",   # AbbVie  (spin-off ABT 2013, cotiza como ABBV desde ene 2013)
    "MRK",    # Merck
    "PFE",    # Pfizer
    "TMO",    # Thermo Fisher
    "ABT",    # Abbott Laboratories
    "DHR",    # Danaher
    "BMY",    # Bristol-Myers Squibb
    "CVS",    # CVS Health
    "CI",     # Cigna
    "ELV",    # Elevance Health  (ex Anthem, rebrand 2022, datos históricos OK)
    "ISRG",   # Intuitive Surgical
    "MDT",    # Medtronic
    "AMGN",   # Amgen
]

# ── Consumer Discretionary ────────────────────────────────────────────────────
CONSUMER_DISC = [
    "AMZN",   # Amazon
    "TSLA",   # Tesla  (IPO 2010)
    "HD",     # Home Depot
    "MCD",    # McDonald's
    "NKE",    # Nike
    "SBUX",   # Starbucks
    "LOW",    # Lowe's
    "BKNG",   # Booking Holdings  (ex Priceline, BKNG desde 2018; yfinance ajusta)
    "TGT",    # Target
    "F",      # Ford
]

# ── Consumer Staples ──────────────────────────────────────────────────────────
CONSUMER_STAPLES = [
    "COST",   # Costco
    "WMT",    # Walmart
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    "MO",     # Altria
    "PM",     # Philip Morris
]

# ── Energy ────────────────────────────────────────────────────────────────────
ENERGY = [
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "COP",    # ConocoPhillips
    "SLB",    # Schlumberger
    "EOG",    # EOG Resources
    "MPC",    # Marathon Petroleum
    "PSX",    # Phillips 66
    "VLO",    # Valero Energy
]

# ── Industrials ───────────────────────────────────────────────────────────────
INDUSTRIALS = [
    "CAT",    # Caterpillar
    "BA",     # Boeing
    "HON",    # Honeywell
    "UPS",    # UPS
    "GE",     # GE Aerospace  (ticker continuo; spin-offs no afectan historia)
    "RTX",    # Raytheon  (ex UTC, RTX desde 2020; yfinance ajusta)
    "LMT",    # Lockheed Martin
    "DE",     # Deere & Co
    "MMM",    # 3M
    "UNP",    # Union Pacific
    "FDX",    # FedEx
    "ETN",    # Eaton
]

# ── Utilities ─────────────────────────────────────────────────────────────────
UTILITIES = [
    "NEE",    # NextEra Energy
    "DUK",    # Duke Energy
    "SO",     # Southern Company
    "AEP",    # American Electric Power
    "EXC",    # Exelon
]

# ── Materials ─────────────────────────────────────────────────────────────────
MATERIALS = [
    "APD",    # Air Products
    "SHW",    # Sherwin-Williams
    "ECL",    # Ecolab
    "NEM",    # Newmont
]

# ── Communication Services ────────────────────────────────────────────────────
COMM = [
    "T",      # AT&T
    "VZ",     # Verizon
    "DIS",    # Disney
    "CMCSA",  # Comcast
]

# ── Universo completo ─────────────────────────────────────────────────────────
UNIVERSE: list[str] = (
    TECH
    + SEMIS
    + FINANCIALS
    + HEALTHCARE
    + CONSUMER_DISC
    + CONSUMER_STAPLES
    + ENERGY
    + INDUSTRIALS
    + UTILITIES
    + MATERIALS
    + COMM
)

# ── Mapa sector → tickers (útil para análisis por sector) ────────────────────
SECTORS: dict[str, list[str]] = {
    "Technology":            TECH,
    "Semiconductors":        SEMIS,
    "Financials":            FINANCIALS,
    "Healthcare":            HEALTHCARE,
    "Consumer Discretionary": CONSUMER_DISC,
    "Consumer Staples":      CONSUMER_STAPLES,
    "Energy":                ENERGY,
    "Industrials":           INDUSTRIALS,
    "Utilities":             UTILITIES,
    "Materials":             MATERIALS,
    "Communication":         COMM,
}

if __name__ == "__main__":
    print(f"Universo RMT: {len(UNIVERSE)} acciones")
    for sector, tickers in SECTORS.items():
        print(f"  {sector:<26} {len(tickers):>3}  {tickers}")
