# Contrato de integración: `run_cpcv` ↔ `RMTStrategy`

Este documento especifica **exactamente** qué espera el lado RMT del backtester
CPCV. Cuando el colaborador entregue la implementación real, comparar su
output con este contrato punto por punto.

---

## 1. Firma de la función

```python
def run_cpcv(
    StrategyClass,          # clase (no instancia) con interfaz get_weights + reset
    data:         pd.DataFrame,   # precios reales (ver §3)
    grid:         list[dict],     # grilla de hiperparámetros (ver §4)
    n_groups:     int   = 10,     # número de grupos CPCV
    n_test:       int   = 2,      # grupos usados como OOS por split
    commissions:  float = 0.0001, # comisión por operación (fracción del valor)
) -> dict:
    ...
```

---

## 2. Dict de retorno esperado

```python
{
    "metrics": {
        "sharpe_mean":      float,   # Sharpe promedio entre paths OOS
        "sharpe_std":       float,   # desviación estándar del Sharpe entre paths
        "psr":              float,   # Probabilistic Sharpe Ratio  (0 < psr < 1)
        "dsr":              float,   # Deflated Sharpe Ratio       (0 < dsr < 1)
        "n_paths":          int,     # número de paths combinatorios generados
        "consensus_params": dict,    # hiperparámetros óptimos encontrados por CPCV
                                     # ej: {"entry_threshold": 2.0, "exit_threshold": 0.5, ...}
    },
    "equity_curves": pd.DataFrame,
    # columnas : "Path_1", "Path_2", ..., "Path_n"   (una por path OOS)
    # índice   : DatetimeIndex con fechas de trading  (preferido)
    #            o RangeIndex numérico 0..T-1         (aceptado, se convierte)
    # valores  : equity en $ (ej: empieza en ~100_000)
}
```

El validador en `scripts/run_validation.py` → `validar_resultado()` verifica
esta estructura automáticamente al conectar el backtester real. Si algo no
coincide, lanza un `AssertionError` con mensaje descriptivo.

---

## 3. Contrato sobre `data`

`data` es el DataFrame de precios que se pasa a `run_cpcv`.

| Propiedad | Valor esperado |
|-----------|----------------|
| Tipo | `pd.DataFrame` |
| Índice | `DatetimeIndex` con días hábiles |
| Columnas | tickers del universo (strings, ej: `"AAPL"`) |
| Valores | precios de cierre ajustados (splits + dividendos) |
| NaN | ninguno — el ingest aplica ffill y dropea tickers >20% NaN |
| Período | 2015-01-02 en adelante, ~2864 filas × 100 columnas |

---

## 4. Contrato sobre `grid`

`grid` es la lista devuelta por `RMTStrategy().param_grid()`. El backtester
la itera; la estrategia es dueña de su significado.

```python
# Formato actual — 12 combinaciones (3 entry × 2 exit × 2 sizing):
[
    {"entry_threshold": 1.5, "exit_threshold": 0.5, "ventana_betas": 252,
     "ventana_zscore": 252, "sizing_by_zscore": True},
    {"entry_threshold": 1.5, "exit_threshold": 0.5, "ventana_betas": 252,
     "ventana_zscore": 252, "sizing_by_zscore": False},
    {"entry_threshold": 1.5, "exit_threshold": 1.0, ...},
    ...  # ver RMTStrategy.param_grid() para la lista completa
]
```

El backtester instancia la estrategia como:
```python
strategy = StrategyClass(**params)   # params = un dict de la grid
```

---

## 5. Contrato de no look-ahead — CRÍTICO

### Lo que el backtester DEBE hacer en su loop interno

```python
strategy = StrategyClass(**params)
strategy.reset()                          # limpiar estado entre paths

# positions: dict {ticker: +1/-1} con posiciones actualmente abiertas.
# En la primera barra es {} (vacío); el backtester lo actualiza tras ejecutar.
positions = {}

for j in range(1, len(data) + 1):
    weights = strategy.get_weights(
        data.iloc[:j],               # ← solo pasado, NUNCA data.iloc[:j+k]
        current_positions=positions, # ← posiciones actualmente abiertas
    )
    # ... ejecutar trades, actualizar equity, actualizar positions ...
```

### Lo que el backtester NO debe hacer

```python
# MAL — look-ahead total:
weights = strategy.get_weights(data)

# MAL — look-ahead parcial:
weights = strategy.get_weights(data.iloc[:j + k])   # k > 0
```

### Por qué

`get_weights` usa `calcular_residuos_rolling(retornos, ventana=ventana_betas)`
que es O(n²) pero stateless: si recibe más filas de las que corresponden a la
barra `j`, la ventana incluirá días futuros y los resultados estarán
contaminados. Verificado empíricamente: pasar `data.iloc[:j+50]` cambia los
pesos vs. `data.iloc[:j]`.

### Guarda opcional (recomendada durante desarrollo)

```python
weights = strategy.get_weights(
    data.iloc[:j],
    current_positions=positions,
    current_bar_date=data.index[j - 1],   # verifica que el índice[-1] coincida
)
```

Si hay un bug en el loop que pase datos incorrectos, `get_weights` lanza
`ValueError` con la fecha esperada vs. la recibida.

---

## 6. Contrato sobre `StrategyClass`

El backtester recibe la **clase**, no una instancia. La instancia la crea
el backtester para cada combinación de parámetros y cada path.

Métodos que el backtester puede llamar:

| Método | Firma | Cuándo |
|--------|-------|--------|
| `__init__` | `StrategyClass(**params)` | al inicio de cada path |
| `get_weights` | `(prices, current_positions=None, current_bar_date=None) → dict` | en cada barra de rebalanceo |
| `reset` | `() → None` | entre paths del mismo conjunto de params (no-op: stateless) |

### Parámetro `current_positions`

```python
# Formato esperado: {ticker: signo}
current_positions = {
    "AAPL":  1,   # long abierto
    "MSFT": -1,   # short abierto
    # tickers sin posición → no incluir (o incluir con valor 0, equivalente)
}
```

- Valores válidos: `+1` (long), `-1` (short).
- El backtester maneja las posiciones en shares/dólares; debe convertir a signo
  antes de pasarlas: `signo = int(np.sign(pos_en_shares))`.
- `None` equivale a `{}` (primera barra, sin posiciones abiertas).
- Si se omite, la estrategia asume que no hay posiciones previas (más conservador:
  no mantiene nada, recalcula desde cero).

### Lógica de entry/exit

- **Entry**: si no hay posición en el ticker y `|z| > entry_threshold` →
  abre long (`z < −threshold`) o short (`z > +threshold`).
- **Exit**: si hay posición y `|z| < exit_threshold` → cierra (mean reversion).
  Si `|z| ≥ exit_threshold` → mantiene (independientemente del signo del z).
- Exit ocurre en la próxima barra de rebalanceo; el backtester ejecuta las
  diferencias de pesos como órdenes.

### `get_weights` devuelve `{ticker: peso}` donde:

- pesos positivos = long
- pesos negativos = short
- peso = 0 = sin posición
- **`Σ|peso_i| = 1`** siempre que haya al menos una posición abierta
- equiponderado (`sizing_by_zscore=False`): `peso_i = ±1/N`
- por z-score (`sizing_by_zscore=True`): `peso_i = ±|z_i| / Σ|z_j|`

---

## 7. Checklist de integración

Al recibir el `run_cpcv` real, verificar en orden:

- [ ] Importar desde `validation/cpcv.py` en lugar de `stub_cpcv.py`
- [ ] El loop interno pasa `data.iloc[:j]` en el paso `j` (no `data` completo)
- [ ] El loop construye `current_positions = {ticker: sign}` y lo pasa en cada barra
- [ ] Correr `run_validation.py` → `validar_resultado()` no lanza errores
- [ ] El plot muestra N paths con equity curves reales (no sintéticas)
- [ ] El eje X tiene fechas reales (DatetimeIndex del resultado)
- [ ] `consensus_params` contiene claves válidas para `RMTStrategy.__init__`,
      incluyendo `sizing_by_zscore`
- [ ] Verificar en el código del backtester que usa `data.iloc[:j]` en el loop
- [ ] Opcional: activar `current_bar_date` en 10 barras y confirmar que no lanza error
