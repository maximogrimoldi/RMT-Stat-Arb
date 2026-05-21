# Backtesting Engine

Motor de backtesting event-driven, agnóstico a la estrategia, diseñado para validar estrategias cuantitativas con rigor estadístico.

---

## Arquitectura

```
DataHandler → MarketEvent → Strategy → SignalEvent → Portfolio → OrderEvent → ExecutionHandler → FillEvent
                 ↑_______________________________________________________________|
```

| Componente | Responsabilidad |
|---|---|
| `DataHandler` | Guardián del tiempo. Barrera estricta: imposible entregar datos del futuro. |
| `Strategy` | Recibe datos, devuelve señales. No sabe en qué fold está. |
| `Portfolio` | Position sizing y filtros de riesgo. No conoce la estrategia. |
| `ExecutionHandler` | Ejecución simulada con slippage y comisiones. Reemplazable por broker real. |

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
    bars_per_year: int = 252           # 252 diario | 52 semanal | 12 mensual
    block_bootstrap_reps: int = 0      # 0 = off, 10_000 = recomendado
    alpha_halflife_bars: int | None = None
    label_horizon: int = 1             # barras de purging
    embargo_pct: float = 0.01
    n_trials: int = 1                  # para DSR
```

---

## CPCVEngine

Responde: *¿qué distribución de resultados puedo esperar?*

Genera `C(N,k)` combinaciones de `k` grupos de test. Reconstruye `φ = C(N−1, k−1)` trayectorias OOS independientes.

```python
CPCVConfig(n_groups=6, n_test_groups=2)
# → C(6,2) = 15 backtests, φ = 5 trayectorias
```

**Reporta**: `sharpes_per_path` (φ valores), media/std/p5 del Sharpe, `pct_positive_paths`, `sharpe_avg_path`, `psr_avg_path`, `max_drawdown`, DSR.

## Tuning interno

`validation/tuning.py` agrega el paso de seleccion de parametros:

- `tune_inner_is_segments(...)` usa solo `is_segments`;
- `build_nested_cpcv_runner(...)` ejecuta el tuning, el fit externo y el predict externo;
- `tune_flat_dataset(...)` queda como ruta de deployment para extraer parametros sobre todo el dataset;
- la salida de tuning no reemplaza al backtest final.

---

## Módulos experimentales

Hay archivos que existen en `validation/` pero no forman parte del pipeline activo:

- `stress_testing.py`: capa genérica de stress testing sobre `BacktestRunner`, útil para shocks de data y PnL.

Hoy esos módulos no se ejecutan desde el flujo principal y no afectan la validación CPCV.

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

- El `DataHandler` mantiene cursor temporal: imposible solicitar datos futuros.
- Features calculados sobre ventanas cerradas (bar `t` usa datos hasta `t−1`).
- Parámetros del estimador congelados al final del IS.
- Scaler/normalizer fiteado solo sobre IS.
- Lookups externos indexados al día anterior.

---

## Estructura de archivos

```
├── validation/
│   ├── config.py          # ValidationConfig
│   ├── cpcv.py            # CPCVConfig + CPCVEngine
│   ├── tuning.py          # Grid tuning + consenso entre folds
│   ├── metrics.py         # Sharpe, PSR, DSR, bootstrap, max_drawdown
│   ├── plots.py           # plot_equity_vs_benchmark
│   ├── stress_testing.py   # Stress testing genérico sobre BacktestRunner
│   └── report.py          # ValidationReport + plot_vs_spy()
├── engine/
│   ├── events.py
│   ├── data_handler.py
│   ├── portfolio.py
│   ├── execution_handler.py
│   └── event_loop.py
└── examples/
    ├── __init__.py
    └── production_tuning.py  # script generico para tuning de deployment
```
