# Backtester — RMT Stat-Arb

Motor de backtesting event-driven con validación CPCV (Combinatorial Purged Cross-Validation) y estrategia de arbitraje estadístico basada en Random Matrix Theory.

Proyecto compartido: Maximo Grimoldi (motor CPCV) + Juan Cruz Albamonte (estrategia RMT).

---

## Estructura

```
Backtester/
├── cpcv/                        # Motor de validación (Maxi)
│   ├── pipeline/                #   CPCVEngine, splits, tuning, config
│   ├── engine/                  #   EventLoop, Portfolio, ExecutionHandler
│   ├── analysis/                #   Métricas: Sharpe, DSR, PSR, drawdown, bootstrap
│   ├── strategy/                #   Interfaz base + EventDrivenEstimator
│   ├── tests/                   #   Tests del motor
│   └── examples/                #   Ejemplos de uso
│
└── rmt_stat_arb/                # Estrategia RMT Stat-Arb (Juan)
    ├── strategy/                #   RMTStrategy (core.py) + pipeline RMT (signals.py)
    ├── data/                    #   Ingest de precios, universo de 100 tickers
    ├── engines/                 #   PaperEngine (paper trading) + IBKRClient (live)
    ├── monitoring/              #   Health checks pre-trade
    ├── scripts/
    │   ├── run_validation_rmt.py  # Backtest CPCV completo
    │   ├── run_paper.py           # Paper trading orquestador
    │   └── smoke_test_paper.py    # Tests del flujo de paper trading
    └── results/                 #   Outputs del backtest (generados al correr)
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Correr el backtest CPCV

```bash
source .venv/bin/activate
python rmt_stat_arb/scripts/run_validation_rmt.py
```

**Outputs generados en `rmt_stat_arb/results/`:**

| Archivo | Contenido |
|---|---|
| `equity_curves.parquet` | 5 paths OOS × ~2860 barras |
| `best_params.json` | Parámetros consensuados para paper trading |
| `metrics.json` | Sharpe, DSR, drawdown, alpha/beta, consensus por fold, grid |
| `figures/equity_curves.png` | Plot de las 5 curvas de equity |

---

## Resultados — corrida final (embargo LdP-compliant)

Configuración: CPCV N=6, k=2 → φ=5 paths. Universo: 100 tickers S&P500. Período: 2015–2026. Embargo: 25 barras (≈1% del dataset, per LdP AFML cap.12).

| Métrica | Valor |
|---|---|
| Sharpe medio | −0.108 |
| Sharpe std | 0.159 |
| DSR | 0.000 |
| Max Drawdown | −38.44% |
| % paths positivos | 40% |
| Alpha anualizado | −0.013 |
| Beta vs S&P500 | 0.034 |

**Parámetros óptimos (consenso de 30 folds):**
`entry_threshold=2.497`, `exit_threshold=1.0`, `sizing_by_zscore=True`, `ventana=252 días`.

---

## Arquitectura técnica

### Motor CPCV (`cpcv/`)

Pipeline de validación con dos capas:

1. **CPCV externo** — C(6,2)=15 combinaciones de splits, reconstruye φ=5 trayectorias OOS independientes.
2. **Tuning interno** — dentro de cada IS, grid search con 4 splits cronológicos internos. Consenso ponderado por recencia (`half_life=365 días`).

### Estrategia RMT (`rmt_stat_arb/`)

Stateless: `get_weights(prices, positions) → {ticker: weight}`. Mismos inputs → mismos outputs siempre.

Pipeline matemático:
1. Retornos diarios → matriz de correlación (ventana rolling 252 días)
2. PCA + filtro Marchenko-Pastur → elimina autovalores bajo λ_max (ruido Wishart)
3. Regresión → residuos idiosincrásicos por ticker
4. Z-score sobre residuos acumulados → señal de reversión a la media
5. Entry si `|z| > threshold`, exit si `|z| < exit_threshold`

Precompute: `calcular_residuos_rolling` corre **una sola vez** sobre el dataset completo antes del CPCV. Los slices temporales usan la matriz pre-computada, eliminando el warmup por fold.

### Nota de imports

`rmt_stat_arb/strategy/` y `cpcv/strategy/` comparten el mismo nombre de paquete. `run_validation_rmt.py` resuelve la colisión con un inject mínimo de `importlib` para `strategy.estimator` y `strategy.base` desde `cpcv/`. Ver comentarios en el script.

---

## Tests

```bash
cd cpcv && python -m pytest tests/ -v
```
