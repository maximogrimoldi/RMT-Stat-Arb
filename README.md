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

## Primer uso (end-to-end)

Después de instalar, los pasos para arrancar de cero:

```bash
# 1. Validar la estrategia (genera best_params.json)
python -m rmt_stat_arb backtest

# 2. Abrir TWS de Interactive Brokers en modo paper trading (puerto 7497)

# 3. Ejecutar el primer rebalanceo paper
python -m rmt_stat_arb paper

# 4. Monitorear el estado en cualquier momento
python -m rmt_stat_arb status
```

## CLI — Tabla de comandos

| Quiero...                                   | Comando                               | Qué hace                                                                                          |
|---------------------------------------------|---------------------------------------|---------------------------------------------------------------------------------------------------|
| **Ejecutar un backtest**                    | `python -m rmt_stat_arb backtest`     | Corre la validación CPCV completa (≈30 segundos). Genera `metrics.json`, `equity_curves.parquet`, `best_params.json` y el plot. |
| **Ver qué activos opera la estrategia**     | `python -m rmt_stat_arb universe`     | Lista los 100 tickers del S&P 500 que opera la estrategia y el rango de fechas de datos disponibles. |
| **Conectarse al broker y operar (paper)**   | `python -m rmt_stat_arb paper`        | Lee `best_params.json`, conecta a TWS, calcula pesos, pide confirmación y ejecuta órdenes paper. |
| **Operar más de una vez al mismo día**      | `python -m rmt_stat_arb paper --force`| Mismo que `paper` pero saltea el check de idempotencia.                                          |
| **Monitorear el portafolio en tiempo real** | `python -m rmt_stat_arb status`       | Muestra capital actual, retorno acumulado, drawdown del mes y posiciones long/short activas.     |

## Detalle de cada comando

### `backtest`

Corre el motor CPCV de López de Prado con purge + embargo, calcula métricas (Sharpe, DSR, drawdown, turnover, alpha/beta vs S&P 500) y aplica 4 escenarios de stress testing sobre el último fold OOS.

Outputs en `rmt_stat_arb/results/backtesting/`:
- `metrics.json` — todas las métricas + diagnóstico del grid + resultados del stress
- `equity_curves.parquet` — 5 trayectorias OOS reconstruidas (CPCV con n=6, k=2)
- `best_params.json` — parámetros consensuados (usados por `paper`)
- `figures/equity_curves.png` — plot con mean ± banda P10-P90

### `paper`

Antes de operar corre 3 checks (datos frescos, TWS conectado en `127.0.0.1:7497`, idempotencia diaria). Lee los parámetros validados, calcula el rebalanceo y pide confirmación humana antes de mandar órdenes.

NAV trackeado con marking-to-market — aísla la sub-estrategia RMT del NAV total de IBKR (necesario si hay otras estrategias corriendo en la misma cuenta paper).

Outputs en `rmt_stat_arb/results/trading/`:
- `daily_state.parquet` — historial de runs (append-only): NAV, pesos, posiciones, z-scores
- `orders_log.parquet` — historial de órdenes BUY/SELL ejecutadas

Post-trade: 4 health checks (market-neutral, gross exposure, n posiciones, z-scores extremos) como warnings.

### `status`

Lectura read-only de `daily_state.parquet`. No conecta a IBKR. Muestra: último run, capital, retorno acumulado vs capital inicial, drawdown del mes, y posiciones long/short con sus pesos.

### `universe`

Lectura read-only del archivo de universo (`data/universe.py`) y de `data/storage/prices.parquet`. Lista los 100 tickers ordenados alfabéticamente en columnas y muestra el rango de fechas de datos disponibles localmente.

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
