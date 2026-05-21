from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl

from validation.config import ValidationConfig
from validation.tuning import tune_flat_dataset


def load_dataset(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset: {path}")

    if path.suffix.lower() in {".parquet", ".pq"}:
        data = pl.read_parquet(path)
    else:
        data = pl.read_csv(path)

    if "timestamp" not in data.columns:
        raise ValueError("El dataset debe contener al menos una columna 'timestamp'.")

    return data.sort("timestamp")


def load_callable(spec: str) -> Callable[..., Any]:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError("El factory debe tener formato 'modulo:callable'.")

    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    if not callable(factory):
        raise TypeError(f"{spec} no apunta a un callable valido.")
    return factory


def parse_grid(value: str) -> list[dict[str, Any]]:
    grid = json.loads(value)
    if not isinstance(grid, list) or not all(isinstance(item, dict) for item in grid):
        raise ValueError("La grilla debe ser una lista JSON de diccionarios.")
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flat tuning generico para parametros de produccion.")
    parser.add_argument("--data-path", required=True, help="Path al dataset completo (parquet o csv).")
    parser.add_argument(
        "--runner-factory",
        required=True,
        help="Callable 'modulo:factory' que recibe params y devuelve un BacktestRunner.",
    )
    parser.add_argument(
        "--grid-json",
        required=True,
        help="Grilla JSON de hiperparametros. Ej: '[{\"alpha\": 10}, {\"alpha\": 20}]'.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Cantidad de splits cronologicos.")
    parser.add_argument("--bars-per-year", type=int, default=252, help="Barras por ano para el scoring.")
    parser.add_argument("--label-horizon", type=int, default=1, help="Horizonte de purging.")
    parser.add_argument("--embargo-pct", type=float, default=0.01, help="Embargo porcentual.")
    parser.add_argument("--embargo-bars", type=int, default=None, help="Embargo absoluto en barras.")
    parser.add_argument("--n-trials", type=int, default=1, help="Cantidad de trials para DSR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(Path(args.data_path))
    runner_factory = load_callable(args.runner_factory)
    grid = parse_grid(args.grid_json)

    val_cfg = ValidationConfig(
        bars_per_year=args.bars_per_year,
        label_horizon=args.label_horizon,
        embargo_pct=args.embargo_pct,
        embargo_bars=args.embargo_bars,
        n_trials=args.n_trials,
    )

    tuning = tune_flat_dataset(
        data=data,
        val_cfg=val_cfg,
        grid=grid,
        runner_factory=runner_factory,
        n_splits=args.n_splits,
    )

    print(f"[+] Parametros optimos para produccion hoy: {tuning.consensus_params}")


if __name__ == "__main__":
    main()
