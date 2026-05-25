# AI Log — Backtester

Registro de cómo se usó AI en el desarrollo del engine de backtesting.
No cubre la estrategia, solo el motor de validación.

---

## Tareas donde la AI fue útil

- Implementar clases que yo ya había diseñado (Portfolio, EventLoop, CPCVEngine, EventDrivenEstimator)
- Sintaxis de Polars (slices, joins, pct_change, drop_nulls)
- Implementar fórmulas estadísticas ya conocidas: PSR, DSR, block bootstrap, expected_max_sharpe
- Boilerplate de ABCs y dataclasses
- Búsqueda de bugs cuando le pedí análisis crítico explícito

---

## Tareas donde decidí no confiar en la AI

- Arquitectura general del engine (event-driven, qué componentes existen y cómo se conectan)
- Decisión de eliminar DataHandler, SignalEvent, MarketEvent, PositionSizer — simplificar a weights directos
- Diseño del contrato de la estrategia: `get_weights(data, positions) → {symbol: weight}`
- Qué valores tiene el dict de posiciones ({1, -1, 0}, no cantidades reales)
- Cómo reconstruir trayectorias CPCV y qué métricas reportar
- Decisión de usar purge + embargo en los splits

---

## Errores de la AI por fase

### Engine (sesiones previas)

**DSR siempre daba ~0**: `deflated_sharpe_ratio` llamaba a `expected_max_sharpe(n_trials)` y multiplicaba el resultado por `sqrt(252)`. El error era asumir que el output de `expected_max_sharpe` estaba en escala per-bar. Con prior `sr_std=1.0` per-bar, el SR* anualizado resultante era ~13.5 — imposible de superar con ninguna estrategia real. Fix: el prior `sr_std=1.0` se interpreta en escala anualizada y el output de `expected_max_sharpe` ya es anualizado, sin multiplicar por `sqrt(252)`.

### Métricas de mercado (sesión actual)

**Information Ratio siempre daba 0**: la fórmula usaba `residuals.mean() / residuals.std()`. Por construcción del OLS, `residuals.mean() == 0` siempre. Fix: `IR = intercept * sqrt(bars_per_year) / tracking_error` (alpha anualizado sobre tracking error anualizado).

**Regresión vs mercado con alineación temporal incorrecta**: la primera implementación regresaba `avg_returns` (promedio barra-a-barra de los 5 paths CPCV) contra los retornos del mercado. El problema: cada path cubre períodos distintos, al promediarlos la barra 1 del path A (2018-Q1) y la barra 1 del path B (2020-Q3) se suman — la serie resultante no corresponde a ninguna fecha real. Beta daba 0.018 y R² ≈ 0 para un portafolio long-only de AAPL/MSFT/GOOGL/AMZN, lo cual es absurdo. Fix: regresión por path individual (cada path sí tiene fechas contíguas válidas), join por timestamp con el SP500, y promedio de alpha/beta/R²/IR entre paths.

---

## Decisiones de diseño que tomé yo, no la AI

- **Sin DataHandler**: en una arquitectura clásica, el DataHandler es el guardián del tiempo. Decidí eliminarlo y que EventLoop controle el cursor temporal directamente con `segment.slice(0, i+1)`. Más simple, menos indirección.

- **Sin SignalEvent**: la estrategia no emite señales que el Portfolio interpreta — devuelve weights directamente. El Portfolio reconcilia target vs posición actual y genera órdenes. Elimina una capa de traducción innecesaria.

- **positions como {1, -1, 0}**: la estrategia recibe un dict de posición actual con valores discretos (long/short/flat), no cantidades reales en unidades. La estrategia no necesita saber cuántas acciones tiene, solo si está long, short o fuera. Las cantidades reales son responsabilidad del Portfolio.

- **IS warmup desechable**: en `EventDrivenEstimator`, el IS warmup corre con un Portfolio separado que se descarta. La estrategia acumula contexto IS, pero el OOS arranca con portfolio fresco. Alternativa hubiera sido pasar directamente el estado IS al OOS — decidí no hacerlo para aislar la contabilidad.

- **Temporal break handling explícito**: cuando hay segmentos IS discontinuos (caso típico en CPCV), `EventLoop` acepta `list[pl.DataFrame]` y cierra todas las posiciones entre segmentos con `force_close()`. Garantiza que no haya posiciones abiertas que crucen períodos temporalmente discontinuos.

- **Consenso de hiperparámetros ponderado por recencia**: en lugar de elegir los mejores hiperparámetros del fold IS más reciente, se promedian ponderando por `exp(-age/half_life_days)`. Reduce sensibilidad a un fold específico.

- **Benchmark automático**: `main()` descarga el SP500 automáticamente para el período del dataset y calcula alpha/beta/R²/IR sin que el usuario tenga que pasarle nada. Si la descarga falla, las métricas se omiten sin romper el backtest.
