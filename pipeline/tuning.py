from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import polars as pl

from pipeline.cpcv import BacktestRunner
from pipeline.config import ValidationConfig
from analysis.metrics import sharpe_ratio
from pipeline.splits import build_train_segments, make_groups


ParamGrid = Sequence[Mapping[str, Any]]
RunnerFactory = Callable[[Mapping[str, Any]], BacktestRunner]
ScoreFn = Callable[[pl.Series, pl.Series], float]


@runtime_checkable
class FitPredictEstimator(Protocol):
    def fit(self, is_segments: list[pl.DataFrame]) -> Any: ...

    def predict(self, oos_data: pl.DataFrame) -> tuple[pl.Series, pl.Series]: ...


EstimatorFactory = Callable[[Mapping[str, Any]], FitPredictEstimator]


@dataclass(frozen=True)
class CandidateResult:
    params: dict[str, Any]
    scores: list[float]
    mean_score: float
    median_score: float
    std_score: float


@dataclass(frozen=True)
class FoldTuningResult:
    combo: tuple[int, ...]
    best_candidate: CandidateResult
    candidates: list[CandidateResult]


@dataclass(frozen=True)
class NestedTuningResult:
    fold_results: list[FoldTuningResult]
    consensus_params: dict[str, Any]


TuningResult = NestedTuningResult


def default_score(returns: pl.Series, signals: pl.Series) -> float:
    del signals
    return sharpe_ratio(returns)


def _clean_scores(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def _summarize_scores(scores: Sequence[float]) -> tuple[float, float, float]:
    clean = _clean_scores(scores)
    if not clean:
        return float("-inf"), float("-inf"), float("inf")
    arr = np.asarray(clean, dtype=float)
    return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _select_best_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("No hay candidatos para seleccionar.")
    return max(
        candidates,
        key=lambda c: (c.median_score, c.mean_score, -c.std_score),
    )


def _inner_splits(data: pl.DataFrame, n_splits: int) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    if n_splits < 2:
        raise ValueError("n_splits debe ser >= 2.")
    groups = make_groups(data, n_splits)
    folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    for i in range(1, n_splits):
        train_parts = [g for g in groups[:i] if len(g) > 0]
        test_part = groups[i]
        if not train_parts or len(test_part) == 0:
            continue
        train = pl.concat(train_parts)
        folds.append((train, test_part))
    return folds


def _evaluate_grid(
    train_segs: list[pl.DataFrame],
    test_part: pl.DataFrame,
    param_grid: ParamGrid,
    runner_factory: RunnerFactory,
    score_fn: ScoreFn,
) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    for params in param_grid:
        runner = runner_factory(params)
        try:
            returns, signals = runner(train_segs, test_part)
            score = float(score_fn(returns, signals))
        except Exception:
            score = float("nan")
        candidates.append(
            CandidateResult(
                params=dict(params),
                scores=[score],
                mean_score=score,
                median_score=score,
                std_score=0.0,
            )
        )
    return candidates


def _decay_weights(fold_end_dates: Sequence[Any], half_life_days: float) -> list[float]:
    dates = list(fold_end_dates)
    max_date = max(dates)
    return [2.0 ** (-((max_date - d).days) / half_life_days) for d in dates]


def _consensus_params(
    winners: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> dict[str, Any]:
    if not winners:
        return {}

    w_arr = np.asarray(weights, dtype=float)
    keys = sorted({key for winner in winners for key in winner.keys()})
    result: dict[str, Any] = {}

    for key in keys:
        idx = [i for i, winner in enumerate(winners) if key in winner]
        values = [winners[i][key] for i in idx]
        w_key = w_arr[idx]
        if not values:
            continue

        if all(_is_numeric(v) for v in values):
            v_arr = np.asarray(values, dtype=float)
            w_norm = w_key / w_key.sum()
            weighted_mean = float(np.dot(w_norm, v_arr))
            if all(float(v).is_integer() for v in values):
                result[key] = int(round(weighted_mean))
            else:
                result[key] = weighted_mean
            continue

        wt: dict[Any, float] = {}
        for v, w in zip(values, w_key.tolist()):
            wt[v] = wt.get(v, 0.0) + w
        result[key] = max(wt, key=lambda k: wt[k])

    return result


def tune_grid_on_is(
    is_data: pl.DataFrame,
    param_grid: ParamGrid,
    runner_factory: RunnerFactory,
    *,
    n_inner_splits: int = 4,
    score_fn: ScoreFn = default_score,
) -> FoldTuningResult:
    """
    Evalua una grilla dentro de un unico IS usando splits cronologicos internos.
    Devuelve el mejor candidato y la tabla completa de resultados.
    """
    folds = _inner_splits(is_data, n_inner_splits)
    if not folds:
        raise ValueError("No se pudieron construir folds internos para tuning.")

    candidates: list[CandidateResult] = []
    for params in param_grid:
        runner = runner_factory(params)
        scores: list[float] = []
        for train_df, test_df in folds:
            try:
                returns, signals = runner([train_df], test_df)
                scores.append(float(score_fn(returns, signals)))
            except Exception:
                scores.append(float("nan"))

        mean_score, median_score, std_score = _summarize_scores(scores)
        candidates.append(
            CandidateResult(
                params=dict(params),
                scores=scores,
                mean_score=mean_score,
                median_score=median_score,
                std_score=std_score,
            )
        )

    best_candidate = _select_best_candidate(candidates)
    return FoldTuningResult(
        combo=(),
        best_candidate=best_candidate,
        candidates=candidates,
    )


def tune_flat_dataset(
    data: pl.DataFrame,
    val_cfg: ValidationConfig,
    grid: list[dict],
    runner_factory: Callable,
    n_splits: int = 5,
) -> TuningResult:
    """
    Tuning flat sobre el dataset completo.
    Divide la serie en bloques cronologicos, usa cada bloque como holdout
    una vez, y agrega los winners ponderados por recencia (val_cfg.half_life_days).
    """
    if n_splits < 2:
        raise ValueError("n_splits debe ser >= 2.")

    groups = make_groups(data, n_splits)
    split_results: list[FoldTuningResult] = []
    winners: list[Mapping[str, Any]] = []
    fold_end_dates: list[Any] = []

    for split_idx, test_part in enumerate(groups):
        if len(test_part) == 0:
            continue
        train_segs = build_train_segments(
            groups,
            {split_idx},
            val_cfg.label_horizon,
            val_cfg.embargo_pct,
            val_cfg.embargo_bars,
        )
        if not train_segs:
            continue

        candidates = _evaluate_grid(train_segs, test_part, grid, runner_factory, default_score)
        best_candidate = _select_best_candidate(candidates)
        split_results.append(
            FoldTuningResult(
                combo=(split_idx,),
                best_candidate=best_candidate,
                candidates=candidates,
            )
        )
        winners.append(best_candidate.params)
        fold_end_dates.append(test_part["timestamp"].max())

    if not split_results:
        raise ValueError("No se pudieron evaluar splits validos para flat tuning.")

    weights = _decay_weights(fold_end_dates, val_cfg.half_life_days)
    return TuningResult(
        fold_results=split_results,
        consensus_params=_consensus_params(winners, weights),
    )


def _fit_estimator(estimator: FitPredictEstimator, is_segments: list[pl.DataFrame]) -> FitPredictEstimator:
    fitted = estimator.fit(is_segments)
    return fitted if fitted is not None else estimator


def _inner_blocks_from_is_segments(is_segments: list[pl.DataFrame]) -> list[pl.DataFrame]:
    return [seg for seg in is_segments if len(seg) > 0]


def tune_inner_is_segments(
    is_segments: list[pl.DataFrame],
    val_cfg: ValidationConfig,
    grid: list[dict],
    estimator_factory: EstimatorFactory,
    n_splits: int = 5,
    score_fn: ScoreFn = default_score,
) -> TuningResult:
    """
    Acto 1: tuning interno usando exclusivamente los bloques IS.
    Cada bloque se trata como una unidad cronologica y se usa purge/embargo
    al armar los train segments del inner split. Winners ponderados por recencia.
    """
    groups = _inner_blocks_from_is_segments(is_segments)
    if len(groups) < 2:
        raise ValueError("Se necesitan al menos 2 bloques IS para el tuning interno.")

    n_inner = min(n_splits, len(groups))
    if n_inner < 2:
        raise ValueError("n_splits debe ser >= 2.")

    fold_results: list[FoldTuningResult] = []
    winners: list[Mapping[str, Any]] = []
    fold_end_dates: list[Any] = []

    for fold_idx in range(n_inner):
        test_part = groups[fold_idx]
        train_segs = build_train_segments(
            groups,
            {fold_idx},
            val_cfg.label_horizon,
            val_cfg.embargo_pct,
            val_cfg.embargo_bars,
        )
        if not train_segs or len(test_part) == 0:
            continue

        candidates: list[CandidateResult] = []
        for params in grid:
            estimator = estimator_factory(params)
            try:
                fitted_estimator = _fit_estimator(estimator, train_segs)
                returns, signals = fitted_estimator.predict(test_part)
                score = float(score_fn(returns, signals))
            except Exception:
                score = float("nan")
            finally:
                del estimator

            candidates.append(
                CandidateResult(
                    params=dict(params),
                    scores=[score],
                    mean_score=score,
                    median_score=score,
                    std_score=0.0,
                )
            )

        best_candidate = _select_best_candidate(candidates)
        fold_results.append(
            FoldTuningResult(
                combo=(fold_idx,),
                best_candidate=best_candidate,
                candidates=candidates,
            )
        )
        winners.append(best_candidate.params)
        fold_end_dates.append(test_part["timestamp"].max())

    if not fold_results:
        raise ValueError("No se pudieron construir folds internos validos.")

    weights = _decay_weights(fold_end_dates, val_cfg.half_life_days)
    return TuningResult(
        fold_results=fold_results,
        consensus_params=_consensus_params(winners, weights),
    )


def build_nested_cpcv_runner(
    val_cfg: ValidationConfig,
    grid: list[dict],
    estimator_factory: EstimatorFactory,
    n_inner_splits: int = 5,
    score_fn: ScoreFn = default_score,
) -> BacktestRunner:
    """
    Construye un BacktestRunner que ejecuta:
      1) tuning interno sobre IS,
      2) fit externo sobre todo el IS con el hiperparametro consenso,
      3) predict ciego sobre OOS.
    """

    def runner(is_segments: list[pl.DataFrame], oos_data: pl.DataFrame) -> tuple[pl.Series, pl.Series]:
        tuning = tune_inner_is_segments(
            is_segments=is_segments,
            val_cfg=val_cfg,
            grid=grid,
            estimator_factory=estimator_factory,
            n_splits=n_inner_splits,
            score_fn=score_fn,
        )
        params = tuning.consensus_params

        estimator = estimator_factory(params)
        try:
            fitted_estimator = _fit_estimator(estimator, is_segments)
            returns, signals = fitted_estimator.predict(oos_data)
            return returns, signals
        finally:
            del estimator
            del tuning

    return runner
