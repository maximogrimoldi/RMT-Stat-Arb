# AI Log 

## 1) Backtester ##

Registro de cómo se usó IA en el desarrollo del engine de backtesting.

---

## Tareas donde la IA fue útil

- Implementar clases que habíamos diseñado (Portfolio, EventLoop, CPCVEngine, EventDrivenEstimator)
- Sintaxis de Polars (slices, joins, pct_change, drop_nulls)
- Implementar fórmulas estadísticas ya conocidas: PSR, DSR, block bootstrap, expected_max_sharpe
- Boilerplate de ABCs y dataclasses
- Búsqueda de bugs cuando pedimos análisis crítico explícito

---

## Tareas donde decidimos no confiar en la IA

- Arquitectura general del engine (event-driven, qué componentes existen y cómo se conectan)
- Decisión de eliminar DataHandler, SignalEvent, MarketEvent, PositionSizer — simplificar a weights directos
- Diseño del contrato de la estrategia: `get_weights(prices, current_positions, current_bar_date, return_diagnostics) → {symbol: weight}`
- Qué valores tiene el dict de posiciones ({1, -1, 0}, no cantidades reales)
- Cómo reconstruir trayectorias CPCV y qué métricas reportar
- Decisión de usar purge + embargo en los splits

---

## Errores de la IA

### Engine (sesiones previas)

**DSR siempre daba ~0**: `deflated_sharpe_ratio` llamaba a `expected_max_sharpe(n_trials)` y multiplicaba el resultado por `sqrt(252)`. El error era asumir que el output de `expected_max_sharpe` estaba en escala per-bar. Con prior `sr_std=1.0` per-bar, el SR* anualizado resultante era ~13.5 — imposible de superar con ninguna estrategia real. Fix: el prior `sr_std=1.0` se interpreta en escala anualizada y el output de `expected_max_sharpe` ya es anualizado, sin multiplicar por `sqrt(252)`.

### Métricas de mercado (sesión actual)

**Information Ratio siempre daba 0**: la fórmula usaba `residuals.mean() / residuals.std()`. Por construcción del OLS, `residuals.mean() == 0` siempre. Fix: `IR = intercept * sqrt(bars_per_year) / tracking_error` (alpha anualizado sobre tracking error anualizado).

**Regresión vs mercado con alineación temporal incorrecta**: la primera implementación regresaba `avg_returns` (promedio barra-a-barra de los 5 paths CPCV) contra los retornos del mercado. El problema: cada path cubre períodos distintos, al promediarlos la barra 1 del path A (2018-Q1) y la barra 1 del path B (2020-Q3) se suman — la serie resultante no corresponde a ninguna fecha real. Beta daba 0.018 y R² ≈ 0 para un portafolio long-only de AAPL/MSFT/GOOGL/AMZN, lo cual es absurdo. Fix: regresión por path individual (cada path sí tiene fechas contíguas válidas), join por timestamp con el SP500, y promedio de alpha/beta/R²/IR entre paths.

---

## Decisiones de diseño que tomamos, no la IA

- **Sin DataHandler**: en una arquitectura clásica, el DataHandler es el guardián del tiempo. Decidimos eliminarlo y que EventLoop controle el cursor temporal directamente con `segment.slice(0, i+1)`. Más simple, menos indirección.

- **Sin SignalEvent**: la estrategia no emite señales que el Portfolio interpreta — devuelve weights directamente. El Portfolio reconcilia target vs posición actual y genera órdenes. Elimina una capa de traducción innecesaria.

- **positions como {1, -1, 0}**: la estrategia recibe un dict de posición actual con valores discretos (long/short/flat), no cantidades reales en unidades. La estrategia no necesita saber cuántas acciones tiene, solo si está long, short o fuera. Las cantidades reales son responsabilidad del Portfolio.

- **IS warmup desechable**: en `EventDrivenEstimator`, el IS warmup corre con un Portfolio separado que se descarta. La estrategia acumula contexto IS, pero el OOS arranca con portfolio fresco. Alternativa hubiera sido pasar directamente el estado IS al OOS — decidimos no hacerlo para aislar la contabilidad.

- **Temporal break handling explícito**: cuando hay segmentos IS discontinuos (caso típico en CPCV), `EventLoop` acepta `list[pl.DataFrame]` y cierra todas las posiciones entre segmentos con `force_close()`. Garantiza que no haya posiciones abiertas que crucen períodos temporalmente discontinuos.

- **Consenso de hiperparámetros ponderado por recencia**: en lugar de elegir los mejores hiperparámetros del fold IS más reciente, se promedian ponderando por `exp(-age/half_life_days)`. Reduce sensibilidad a un fold específico.

- **Benchmark automático**: `main()` descarga el SP500 automáticamente para el período del dataset y calcula alpha/beta/R²/IR sin que el usuario tenga que pasarle nada. Si la descarga falla, las métricas se omiten sin romper el backtest.

---

## 2) Estrategia + Paper Trading + CLI ##

Registro de cómo se usó IA en el desarrollo de la estrategia RMT, el paper trading contra IBKR, y la integración del CLI.

---

## Tareas donde la IA fue útil

- Implementar la matemática de RMT que entendíamos conceptualmente: Marchenko-Pastur, regresión OLS sobre factores, cálculo de residuos rolling.
- Boilerplate de la conexión a IBKR vía `ibapi`: el callback pattern es feo y repetitivo.
- Sintaxis de pandas/polars cuando convivían los dos formatos en distintas partes del pipeline.
- Detectar bugs de magnitud y semántica cuando le pasamos los outputs (slippage que mejoraba pérdidas, NAV calculado sin cash, turnover propagado a 0 por wrappers).

---

## Tareas donde decidimos no confiar en la IA

- Contrato de la estrategia: que `get_weights(prices, current_positions, current_bar_date, return_diagnostics)` sea estrictamente stateless. La IA tendía a sugerir mantener estado entre llamadas — lo bloqueamos porque rompe la validación por CPCV.
- Grid de hiperparámetros: la IA sugería ampliarlo a 12-18 combinaciones. Lo limitamos a 6 (3 entry thresholds × 2 sizing) para mantener CPCV honesto y evitar p-hacking — un grid grande con DSR penaliza el resultado.
- Cuándo ampliar el grid después de ver resultados. Mantuvimos la regla: solo ampliamos si el tuner converge al borde del grid (diagnóstico ex-ante), nunca para mejorar el Sharpe (eso sería p-hacking que el DSR captura).

---

## Errores de la IA por fase

### Paper trading (refactor inicial)

**NAV calculado sin cash**: la primera versión del NAV en `PaperEngine.execute()` sumaba `qty × precio` de las posiciones IBKR. Para una estrategia long/short equilibrada, longs y shorts se cancelan y el resultado da cerca de cero. Fix: cálculo con marking-to-market — `NAV_hoy = NAV_ayer × (1 + Σ peso × ret_diario)`, persistido en `daily_state.parquet`, que aísla el P&L de la sub-estrategia RMT de cualquier otra posición en la cuenta IBKR.


### Stress testing (último fold)

**Slippage al revés**: la función `apply_slippage_bps` del motor usaba `arr - sign(arr) * drag`. Eso suma drag a retornos negativos (porque `-(-1 × drag) = +drag`), por lo que slippage "mejoraba" las pérdidas. Lo detectamos cuando el escenario "Slippage 10x" daba Sharpe positivo, lo cual es físicamente imposible. Fix: `arr - drag_por_barra`, con drag distribuido proporcionalmente entre barras asumiendo 12 trades/año (rebalanceo mensual).


### Validación

**`oos_transform` redundante con precompute**: la IA propuso 5 escenarios de stress incluyendo `volatility_shock(1.5)` y `liquidity_shock(0.7)` que transforman precios. Pero con precompute los residuos RMT se calculan una vez sobre datos originales — escalar precios en OOS no recomputa la dinámica de factores. El efecto neto es matemáticamente equivalente a escalar el PnL. Detectamos la redundancia al razonarlo. Decisión: eliminar esos dos escenarios, quedarnos con 4 PnL-side honestos (slippage 5x/10x, fee drag, PnL crush) y documentar la limitación arquitectónica.

---

## Decisiones de diseño que tomamos, no la IA

- **Contrato stateless absoluto**: `get_weights(prices, current_positions, current_bar_date, return_diagnostics)` no guarda nada entre llamadas. Es lo que valida el CPCV: si la estrategia tuviera estado interno, el resultado dependería del orden en que se ejecutan los folds.

- **Precompute hook para residuos RMT**: en lugar de recalcular `calcular_residuos_rolling` en cada llamada a `get_weights()` (que pasaría 15 combos × 4 inner splits × 6 outer combos × ~2400 barras = millones de cálculos redundantes), definimos `RMTStrategyPolarsPrecomputed.precompute()` que corre una vez sobre todo el dataset y guarda los residuos en un global. Reduce el costo del CPCV de ~100 minutos a ~4 minutos. Causalidad estricta preservada: los residuos del día `t` solo usan datos `[t-252, t)`.


- **Checks**: Cuatro health checks como warnings (market-neutral, gross, n_pos, z-scores extremos).


- **Sin estimación de Ornstein-Uhlenbeck**: el documento de la estrategia originalmente describía Avellaneda-Lee con OU completo (estimar κ, θ, σ por AR(1) y normalizar con σ_eq = σ/√(2κ)). El código nunca hizo eso — usamos z-score empírico desde el principio. En la corrida final mantuvimos la versión simple porque agregar OU no aporta señal en este universo y dispara la complejidad.

- **CLI integrado en vez de scripts sueltos**: dos scripts en `scripts/` (`run_validation_rmt.py`, `run_paper.py`) no transmiten profesionalismo. Armamos `python -m rmt_stat_arb {backtest|paper|status|universe}` con `argparse` para que el sistema se opere desde un punto de entrada único, alineado con lo que pide la consigna sobre framework usable.

- **Marking-to-market vs NetLiquidation de IBKR**: el NAV de IBKR es el de toda la cuenta paper, que puede tener DQI u otras estrategias corriendo en paralelo. Para aislar la sub-estrategia RMT trackeamos NAV internamente vía `daily_state.parquet`. Cada run actualiza con `NAV_hoy = NAV_ayer × (1 + ret_portafolio_diario)`. Funciona con pesos negativos por el dot product con signo.

- **Filtro ADF sobre residuos diarios antes de abrir posición**: la estrategia originalmente asumía mean-reversion implícita — si el z-score era extremo, entraba. Identificamos que eso era metodológicamente incorrecto: no se valida estadísticamente que el residuo sea estacionario antes de apostar a su reversión. Agregamos un filtro: `adfuller(residuos_diarios)` con p < 0.05 como condición de entry. Decisión metodológicamente correcta aunque empeoró ligeramente el Sharpe — preferimos rigor sobre números.
