"""
validation/hamilton.py

Utilidad para tratar series IS discontinuas como espacio de estados con
observaciones faltantes (Hamilton 1994, Cap. 13).

Uso: pasar el DataFrame resultante a statsmodels.tsa.statespace.MLEModel
o cualquier Kalman filter que maneje NaN como missing observations.
"""
from __future__ import annotations

import polars as pl


def build_gapped_is(
    is_segments: list[pl.DataFrame],
    full_data: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Reconstruye la serie IS como un espacio de estados con observaciones faltantes.

    Parámetros
    ----------
    is_segments : list[pl.DataFrame]
        Bloques IS devueltos por build_train_segments() — ya purgados y embargados.
    full_data : pl.DataFrame | None
        Dataset completo (con columna 'timestamp'). Necesario para rellenar
        los gaps con filas NaN. Sin él, solo se devuelven las filas IS.

    Retorna
    -------
    pl.DataFrame
        Mismas columnas que los segmentos IS + columna 'is_observed' (bool).
        - is_observed=True  : observación IS real, aporta a la log-verosimilitud.
        - is_observed=False : barra de gap, solo propaga el estado (NaN en datos).

    Invariantes
    -----------
    - Timestamp monotónico creciente.
    - is_observed=True solo en barras presentes en algún is_segments[i].
    - Ninguna barra del OOS externo entra en el resultado.
    - Con full_data, los gaps son exactamente las barras de full_data dentro del
      rango IS que no pertenecen a ningún segmento.
    """
    if not is_segments:
        return pl.DataFrame()

    is_parts = [
        seg.with_columns(pl.lit(True).alias("is_observed"))
        for seg in is_segments
        if len(seg) > 0
    ]
    if not is_parts:
        return pl.DataFrame()

    is_df = pl.concat(is_parts).sort("timestamp")

    if full_data is None:
        return is_df

    ts_min = is_df["timestamp"].min()
    ts_max = is_df["timestamp"].max()
    is_timestamps = set(is_df["timestamp"].to_list())

    gap_rows = full_data.filter(
        (pl.col("timestamp") >= ts_min)
        & (pl.col("timestamp") <= ts_max)
        & (~pl.col("timestamp").is_in(list(is_timestamps)))
    )

    if len(gap_rows) == 0:
        return is_df

    # Reemplaza todas las columnas de datos con null, conserva timestamp
    null_exprs = [
        pl.lit(None).cast(dtype).alias(col)
        if col != "timestamp"
        else pl.col("timestamp")
        for col, dtype in gap_rows.schema.items()
    ]
    gap_df = gap_rows.select(null_exprs).with_columns(
        pl.lit(False).alias("is_observed")
    )

    return pl.concat([is_df, gap_df]).sort("timestamp")
