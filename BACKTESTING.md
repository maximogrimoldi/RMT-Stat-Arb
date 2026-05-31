# Backtesting Engine

Motor de backtesting event-driven, agnóstico a la estrategia, diseñado para validar estrategias cuantitativas con rigor estadístico.

---

## Arquitectura

```
EventLoop
  └── por cada barra:
        ├── execution_handler.fill_pending()        # fills pendientes del turno anterior
        ├── [si día de rebalanceo]
        │     ├── strategy.get_weights(history, positions)
        │     ├── portfolio.on_weights(weights, prices, timestamp)  → OrderEvent en queue
        │     └── execution_handler.on_order()  →  FillEvent en queue  →  portfolio.on_fill()
        └── [si día normal]
              └── portfolio.update_market(prices, timestamp)
```

| Componente | Responsabilidad |
|---|---|
| `EventLoop` | Cursor temporal. Itera barra a barra, llama a la estrategia en días de rebalanceo. Entre segmentos IS discontinuos, cierra todas las posiciones via `force_close()`. |
| `Strategy` | Recibe el historial hasta la barra actual y el dict de posiciones abiertas. Devuelve `{ticker: peso_objetivo}` con signo (long > 0, short < 0). |
| `Portfolio` | Contabilidad: reconcilia target weights con posición actual, genera OrderEvents, registra fills, expone `equity_curve` y `turnover_acum`. El sizing es directo: `qty = equity × weight / price`. |
| `ExecutionHandler` | Simulación de fills con slippage porcentual, derechos de mercado y arancel ALYC. |

El motor no implementa estrategias: las orquesta y valida.

---

## Pipeline de validación

Nested CPCV en dos capas:

```
Paso 1 — build_nested_cpcv_runner  →  Acto 1 + Acto 2 + Acto 3
Paso 2 — CPCVEngine                →  ¿qué distribución de equity curves esperar?
```

### BacktestRunner — contrato universal

```python
BacktestRunner = Callable[[list[pl.DataFrame], pl.DataFrame], tuple[pl.Series, pl.Series]]
# is_segments : bloques IS con purging/embargo aplicados (lista porque puede ser no-contiguo)
# oos_data    : período de test
# Retorna     : (retornos OOS, señales OOS)
```

### ValidationConfig

```python
@dataclass
class ValidationConfig:
    bars_per_year: int = 252            # 252 diario | 52 semanal | 12 mensual
    block_bootstrap_reps: int = 0       # 0 = off, 10_000 = recomendado
    alpha_halflife_bars: int | None = None
    label_horizon: int = 1              # barras de purging
    embargo_pct: float = 0.01           # fracción del fold a embargar (fallback)
    embargo_bars: int | None = None     # barras absolutas; overridea embargo_pct si se setea
    n_trials: int = 1                   # para DSR
    half_life_days: float = 365.0       # decaimiento para consenso de hiperparámetros
```

---

## CPCVEngine

Responde: *¿qué distribución de resultados puedo esperar?*

Genera `C(N,k)` combinaciones de `k` grupos de test. Reconstruye `φ = C(N−1, k−1)` trayectorias OOS independientes.

```python
CPCVConfig(n_groups=6, n_test_groups=2)
# → C(6,2) = 15 backtests, φ = 5 trayectorias
```

**Reporta**: `sharpes_per_path` (φ valores), media/std/p5 del Sharpe, `pct_positive_paths`, `sharpe_avg_path`, `psr_avg_path`, `max_drawdown`, `annualized_return_avg`, DSR, `turnover_per_path`, `turnover_annual_mean`.

---

## Precompute hook

Para estrategias cuyo cálculo de features es costoso (ej. RMT con descomposición rolling de matrices de correlación), `CPCVEngine.run()` ejecuta un hook opcional `runner.precompute(data)` una sola vez antes de los splits CPCV. El hook puede:

- Pre-calcular features sobre todo el dataset (manteniendo causalidad estricta — la función rolling solo usa datos pasados).
- Almacenar el resultado en un global de módulo o en estado compartido.
- Devolver los datos sin modificar (o transformados) para el motor.

Esto reduce el costo computacional de O(N × M) a O(N + M), donde N = barras y M = combinaciones × folds del CPCV anidado. Se conecta vía `build_nested_cpcv_runner(..., precompute_fn=...)`.

---

## Tuning interno

`pipeline/tuning.py` agrega el paso de selección de parámetros:

- `tune_inner_is_segments(...)` usa solo `is_segments`;
- `build_nested_cpcv_runner(...)` ejecuta el tuning, el fit externo y el predict externo;
- `tune_flat_dataset(...)` queda como ruta de deployment para extraer parámetros sobre todo el dataset;
- la salida de tuning no reemplaza al backtest final.

---

## Métricas

### Sharpe Ratio
Annualizado: `SR = mean(r) / std(r) × √bars_per_year`. En CPCV: por trayectoria y sobre el path promedio.

### Probabilistic Sharpe Ratio (PSR)
```
PSR = Φ[ (SR̂ − SR*) · √(T−1) / √(1 − γ₃·SR̂ + ((γ₄+2)/4)·SR̂²) ]
```
Incorpora skewness y kurtosis reales. PSR > 95%: significativo. PSR < 70%: probablemente ruido.

### Deflated Sharpe Ratio (DSR)
PSR donde `SR*` se eleva al máximo esperado entre `n_trials` intentos. Penaliza p-hacking. Activado cuando `n_trials > 1`.

### Block Bootstrap
Series sintéticas con bootstrap por bloques (longitud = `alpha_halflife_bars`). Preserva autocorrelación. Reporta `{mean, std, p5, p95}`. Si `p5 < 0`: resultado no robusto.

---

## Turnover tracking

`Portfolio` acumula el turnover (`Σ |Δw_t|`) en cada llamada a `on_weights()`, expuesto como `portfolio.turnover_acum`. `EventDrivenEstimator.predict()` lo propaga como `last_turnover`, y `CPCVEngine` lo recoge por path y reporta:

- `turnover_per_path` — lista de turnover acumulado por trayectoria OOS
- `turnover_annual_mean` — turnover promedio anualizado (`Σ|Δw| × bars_per_year / len(returns)`)

Útil para evaluar costos de transacción reales contra los simulados y para detectar overfitting (estrategias con turnover excesivo suelen estar sobre-ajustadas al ruido).

---

## Stress testing

`analysis/stress_testing.py` forma parte del pipeline activo: se ejecuta automáticamente al final de `python -m rmt_stat_arb backtest`, sobre el último fold OOS del CPCV.

Arquitectura genérica sobre `BacktestRunner`:

- `StressScenario(name, is_transform, oos_transform, pnl_transform)` — composición de transformaciones.
- `StressTester.run(is_segments, oos_data, runner, scenarios)` — devuelve `StressReport` con baseline y delta de métricas por escenario.

Escenarios activos (4 PnL-side, agnósticos al modelo):

| Escenario | Mecanismo |
|---|---|
| Slippage 5× (50 bps) | `apply_slippage_bps(50)` — drag por trade distribuido en barras |
| Slippage 10× (100 bps) | `apply_slippage_bps(100)` |
| Fee drag 5 bps/día | `apply_pnl_drag(0.0005)` — drag fijo por barra |
| PnL crush 50% | `scale_pnl(0.5)` — escala retornos (Sharpe invariante) |

Nota: escenarios de vol/correlaciones (volatility_shock, liquidity_shock) están disponibles pero no se ejecutan en el flujo principal porque con precompute los residuos RMT están fijos — el efecto se reduce a escalar el PnL. Requeriría recalibración del modelo factorial.

---

## Checks automáticos

**Pre-flight** (antes de correr):
```
✓ k < N  y  φ >= 2
```

**Post-flight** (después de correr):
```
✓ pct_positive_paths >= 60%
✓ DSR si n_trials > 1
✓ bootstrap p5 si block_bootstrap_reps > 0
```

Severidades: `INFO / WARNING / ERROR`. Solo los errores bloquean la ejecución.

---

## Veredicto de validez

Un resultado es creíble para capital real cuando se cumplen **todas**:

1. Sharpe positivo con PSR > 90%
2. ≥ 60% de paths con Sharpe > 0
3. Block bootstrap p5 > 0
4. DSR > 90% si se exploraron múltiples combinaciones de parámetros

---

## Prohibición de look-ahead bias

- `EventLoop` mantiene cursor temporal: en cada barra `t`, `strategy.get_weights()` recibe `segment.slice(0, t+1)` — solo pasado.
- Features calculados sobre ventanas cerradas (barra `t` usa datos hasta `t−1`).
- Parámetros del estimador congelados al final del IS.
- Scaler/normalizer fiteado solo sobre IS.
- El precompute hook precalcula rolling features con la misma garantía causal (ventana `[t-k, t-1]`).

---

## Estructura de archivos

```
cpcv/
├── pipeline/
│   ├── config.py          # ValidationConfig
│   ├── cpcv.py            # CPCVConfig + CPCVEngine
│   ├── splits.py          # make_groups, build_train_segments (purging/embargo)
│   └── tuning.py          # Grid tuning + consenso entre folds + precompute hook
├── analysis/
│   ├── metrics.py         # Sharpe, PSR, DSR, bootstrap, max_drawdown, market_regression
│   ├── report.py          # ValidationReport
│   └── stress_testing.py  # Stress testing genérico sobre BacktestRunner
├── engine/
│   ├── events.py          # OrderEvent, FillEvent
│   ├── portfolio.py       # Portfolio — sizing directo + turnover tracking
│   ├── execution_handler.py  # SimulatedExecutionHandler con slippage/comisiones
│   └── event_loop.py      # Cursor temporal, rebalanceo por frecuencia, force_close entre IS
├── strategy/
│   ├── base.py            # Strategy ABC
│   └── estimator.py       # EventDrivenEstimator — envuelve Strategy como FitPredictEstimator
└── tests/
    ├── test_cpcv.py
    ├── test_metrics.py
    ├── test_stress_testing.py
    └── test_tuning.py
```
