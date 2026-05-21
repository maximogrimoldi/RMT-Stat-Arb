from __future__ import annotations

import polars as pl


def make_groups(data: pl.DataFrame, n_groups: int) -> list[pl.DataFrame]:
    total = len(data)
    size = total // n_groups
    groups: list[pl.DataFrame] = []
    for i in range(n_groups):
        start = i * size
        end = start + size if i < n_groups - 1 else total
        groups.append(data.slice(start, end - start))
    return groups


def build_train_segments(
    groups: list[pl.DataFrame],
    test_set: set[int],
    label_horizon: int,
    embargo_pct: float,
    embargo_bars: int | None,
) -> list[pl.DataFrame]:
    """
    Extrae segmentos de entrenamiento contiguos aplicando purge al grupo previo
    al test y embargo al grupo siguiente al test.
    """
    n = len(groups)
    segments: list[pl.DataFrame] = []
    current: list[pl.DataFrame] = []

    for i in range(n):
        if i in test_set:
            if current:
                last = current[-1]
                purge = label_horizon
                last = last.slice(0, max(0, len(last) - purge))
                current[-1] = last
                merged = pl.concat([g for g in current if len(g) > 0])
                if len(merged) > 0:
                    segments.append(merged)
                current = []
        else:
            g = groups[i]
            if i > 0 and (i - 1) in test_set:
                embargo = (
                    embargo_bars
                    if embargo_bars is not None
                    else max(1, int(len(g) * embargo_pct))
                )
                g = g.slice(min(embargo, len(g)))
            if len(g) > 0:
                current.append(g)

    if current:
        merged = pl.concat([g for g in current if len(g) > 0])
        if len(merged) > 0:
            segments.append(merged)

    return segments
