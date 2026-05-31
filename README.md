# RMT Stat-Arb

Estrategia de Statistical Arbitrage basada en Random Matrix Theory (RMT) sobre 100 acciones líquidas del S&P 500.

Proyecto del curso de Ingeniería Financiera (F414) — Universidad de San Andrés, 2026.

## Estructura del repositorio

```
Backtester/
├── cpcv/                    Motor de backtesting CPCV (Combinatorial Purged Cross-Validation)
└── rmt_stat_arb/            Estrategia RMT
    ├── data/                Ingesta y universo de tickers
    ├── strategy/            Lógica RMT (signals + core)
    ├── engines/             Conexión IBKR + paper trading
    ├── monitoring/          Health checks + comando status
    ├── scripts/             Scripts internos invocados por el CLI
    └── results/
        ├── backtesting/     Outputs del CPCV (equity curves, métricas, mejores parámetros)
        └── trading/         Outputs de paper trading (daily state, orders log)
```

## Instalación

Requiere Python 3.11+ y TWS (Trader Workstation) de Interactive Brokers para paper trading.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso — CLI

El sistema se opera desde un único comando con 3 subcomandos:

### `backtest` — Validación CPCV

Corre la validación completa con CPCV anidado sobre el grid de hiperparámetros, calcula métricas de desempeño, genera el plot de equity curves, y agrega stress testing.

```bash
python -m rmt_stat_arb backtest
```

Outputs en `rmt_stat_arb/results/backtesting/`:
- `equity_curves.parquet` — 5 trayectorias OOS reconstruidas
- `metrics.json` — Sharpe, DSR, drawdown, % paths positivos, alpha/beta, grid, consenso por fold, stress testing
- `best_params.json` — parámetros consensuados (usados después por `paper`)
- `figures/equity_curves.png` — plot de las 5 curvas

### `paper` — Paper trading contra IBKR

Lee los parámetros validados del backtest, conecta a TWS, calcula el rebalanceo y ejecuta órdenes. Pide confirmación humana antes de mandar las órdenes.

```bash
python -m rmt_stat_arb paper
```

Pre-trade checks: datos frescos, TWS conectado, idempotencia (no rebalancea dos veces el mismo día).

Para saltear la idempotencia (correr más de una vez al día):

```bash
python -m rmt_stat_arb paper --force
```

Post-trade: imprime capital actual, drawdown del mes, posiciones activas y 4 health checks (market-neutral, gross exposure, n posiciones, z-scores extremos).

Outputs en `rmt_stat_arb/results/trading/`:
- `daily_state.parquet` — historial completo de runs (append-only)
- `orders_log.parquet` — historial de órdenes BUY/SELL

### `status` — Estado actual del portfolio

Muestra el estado del portfolio sin operar (read-only sobre `daily_state.parquet`).

```bash
python -m rmt_stat_arb status
```

Output: último run, capital actual, retorno acumulado, drawdown del mes, posiciones long y short.

## Flujo de trabajo típico

1. **Validar la estrategia**: `python -m rmt_stat_arb backtest` (~40 segundos)
2. **Operar**: `python -m rmt_stat_arb paper` (con TWS abierto en paper mode, puerto 7497)
3. **Monitorear**: `python -m rmt_stat_arb status` en cualquier momento

## Estrategia

La estrategia opera residuos idiosincrásicos de un modelo factorial RMT:

1. Calcula la matriz de correlación de los retornos diarios.
2. Aplica el umbral de Marchenko-Pastur para separar factores significativos del ruido.
3. Regresiona los retornos contra los factores y obtiene los residuos.
4. Calcula un z-score empírico (media y desvío de la ventana) sobre el residuo acumulado.
5. Abre long si `z < -entry_threshold`, short si `z > +entry_threshold`.
6. Cierra cuando `|z| < exit_threshold` o cuando el z cruza al lado opuesto.

Es **stateless**: `get_weights(prices, current_positions)` devuelve siempre los mismos pesos dados los mismos inputs. Esto es lo que valida la metodología CPCV.

## Referencias

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. (Cap. 7-8, 12: CPCV, DSR, purge + embargo)
- Avellaneda, M., & Lee, J.H. (2010). *Statistical Arbitrage in the U.S. Equities Market*. Quantitative Finance, 10(7).
- Bun, J., Bouchaud, J.P., & Potters, M. (2017). *Cleaning large correlation matrices: tools from Random Matrix Theory*. Physics Reports, 666.
- Cartea, Á., Cucuringu, M., & Jin, Q. (2023). *Correlation Matrix Clustering for Statistical Arbitrage Portfolios*.

## Documentación adicional

- `AI_LOG.md` — Registro del uso de herramientas de IA en el desarrollo del motor
- `BACKTESTING.md` — Arquitectura del motor de backtesting (CPCV)

## Autores

- Juan Albamonte — Estrategia RMT
- Maximiliano Grimoldi — Motor de backtesting CPCV
- Quinto Adoquín — Documentación académica
