# Workflow de backtesting

Este repo valida estrategias con un flujo estricto de Nested CPCV.
La secuencia correcta es:

1. `CPCVEngine` define los outer splits.
2. Dentro de cada `IS` outer, el runner hace tuning interno.
3. Con el hiperparametro consenso, el estimador se ajusta sobre todo el `IS`.
4. El `OOS` solo se usa para inferencia.
5. Los paths finales se ensamblan respetando el orden cronologico.

---

## 1. Entrada

La entrada esperada es un `polars.DataFrame` con:

- `timestamp` como columna temporal;
- `open`, `close` y las columnas extra que use la estrategia;
- features adicionales si queres preprocesarlas antes del backtest.

---

## 2. El contrato del runner

El punto de extension real es el `BacktestRunner`.

```python
BacktestRunner = Callable[[list[pl.DataFrame], pl.DataFrame], tuple[pl.Series, pl.Series]]
```

Recibe:

- `is_segments`: bloques de entrenamiento;
- `oos_data`: bloque out-of-sample.

Devuelve:

- retornos OOS netos de costos;
- señales OOS.

---

## 3. Orquestacion estrica

La funcion que implementa los tres actos es `build_nested_cpcv_runner(...)`.

```python
from validation.config import ValidationConfig
from validation.tuning import build_nested_cpcv_runner

val_cfg = ValidationConfig(
    bars_per_year=252,
    label_horizon=1,
    embargo_pct=0.01,
)

grid = [
    {"alpha": 10, "beta": 1.5, "gamma": 0.5},
    {"alpha": 20, "beta": 2.0, "gamma": 0.5},
    {"alpha": 40, "beta": 2.0, "gamma": 1.0},
]

runner = build_nested_cpcv_runner(
    val_cfg=val_cfg,
    grid=grid,
    estimator_factory=estimator_factory,
    n_inner_splits=5,
)
```

Ese runner hace:

1. tuning interno usando solo `is_segments`;
2. fit externo con todo el `IS` y el hiperparametro consenso;
3. predict ciego sobre `oos_data`.

---

## 4. Ejecucion final

El runner anidado se pasa directo a `CPCVEngine`.

```python
from validation.cpcv import CPCVConfig, CPCVEngine

cpcv_cfg = CPCVConfig(
    n_groups=6,
    n_test_groups=2,
)

report = CPCVEngine(val_cfg, cpcv_cfg).run(data, runner)
print(report.summary())
print(report.metrics)
```

---

## 5. Que hace el tuning interno

`tune_inner_is_segments(...)` dentro de `validation/tuning.py`:

- usa exclusivamente los bloques `is_segments`;
- arma folds internos purgados;
- prueba toda la grilla;
- extrae el mejor hiperparametro por fold;
- devuelve el consenso estadistico de esos winners.

No toca `oos_data`.

---

## 6. Que hace el fit externo

Una vez obtenido el consenso:

- se instancia el estimador con ese hiperparametro fijo;
- se llama `estimator.fit(is_segments)`;
- el estimador calcula sus parametros endogenos usando todo el `IS`;
- esos parametros no se recalibran con el `OOS`.

---

## 7. Que hace el predict externo

Con el estimador ya hidratado:

- se llama `estimator.predict(oos_data)`;
- el bloque OOS permanece ciego;
- la salida son retornos y señales OOS;
- `CPCVEngine` usa esos retornos para reconstruir los paths.

---

## 8. Que hace CPCVEngine

`CPCVEngine`:

- divide la serie en `N` grupos;
- toma combinaciones de `k` grupos de test;
- arma segmentos IS con purging y embargo;
- llama al runner una vez por cada grupo de test de cada combinacion;
- reconstruye `phi = C(N-1, k-1)` trayectorias OOS;
- calcula Sharpe, PSR, DSR, max drawdown y bootstrap.

## 9. Comparacion contra benchmark

Cuando queres inspeccionar visualmente una estrategia, el `ValidationReport`
expone `plot_vs_spy(...)` como salida opcional.

```python
report.plot_vs_spy(
    output="equity_vs_spy.png",
    strategy_label="Mi estrategia",
    benchmark_ticker="SPY",
)
```

Eso compara la curva OOS agregada contra SPY. No cambia la validacion, solo
agrega visualizacion para revisar la estrategia.

---

## 10. Funciones de tuning disponibles

Hay dos niveles:

- `build_nested_cpcv_runner(...)`: flujo estricto de produccion estadistica.
- `tune_flat_dataset(...)`: extraccion de parametros de produccion sobre todo el dataset, sin outer OOS.

La primera es la correcta para Nested CPCV.
La segunda sirve para deployment final, no para validar performance.

---

## 10. Regla practica

Si queres implementar cualquier estrategia seria en este repo:

1. definis una grilla razonable;
2. implementas `estimator_factory(params)`;
3. construis el runner con `build_nested_cpcv_runner(...)`;
4. pasas ese runner a `CPCVEngine`;
5. tomas la decision con el `ValidationReport`.

No se mezcla tuning con evaluacion final.
No se usa informacion del OOS para ajustar el IS.
