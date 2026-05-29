# Diagnóstico del grid — corrida final (embargo_bars=25)

## sizing_by_zscore
- True: 30/30 folds (100%)
- False: 0/30 folds
- Conclusión: dimensión saturada; el sizing proporcional a |z| es preferido unánimemente.

## entry_threshold
- Mínimo: 1.5007
- p25: 2.0150
- Mediana: 2.4974
- p75: 2.5000
- Máximo: 2.5000
- Distribución por bin:
  - ~1.5: 4 folds (13%)
  - ~2.0: 8 folds (27%)
  - ~2.5: 18 folds (60%)
- Conclusión: dispersión genuina entre folds; el grid cubre el rango relevante sin convergencia a un borde. No se justifica ampliarlo.

## Decisión metodológica
Mantener el grid actual. Cualquier ampliación post-resultado constituiría p-hacking, que el DSR está diseñado para penalizar.

## Resultados finales (CPCV con embargo según LdP)
- Sharpe medio: -0.108
- Sharpe std: 0.159
- DSR: 0.000
- Max Drawdown: -38.44%
- % paths positivos: 40%
- Consenso para paper: entry=2.497, exit=1.0, sizing=True, ventanas=252
