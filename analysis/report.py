from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass
class ValidationReport:
    metrics: dict[str, Any] = field(default_factory=dict)
    equity_curves: list[pl.DataFrame] = field(default_factory=list)
    date_range: tuple[str, str] | None = None
