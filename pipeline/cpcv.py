"""
Combinatorial Purged Cross-Validation (CPCV) — AFML Cap. 12.

Ventaja sobre WFA lineal: en lugar de una única trayectoria OOS, genera
φ = C(N−1, k−1) trayectorias independientes, permitiendo estimar la
distribución del Sharpe (media, std, p5) en lugar de solo su valor puntual.

Terminología:
  N = n_groups     : número de grupos en que se divide la serie temporal
  k = n_test_groups: grupos de test por combinación (k < N)
  C(N,k)           : número total de combinaciones (backtests)
  φ = C(N−1,k−1)  : número de trayectorias OOS completas reconstruibles
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations as _combinations
from math import comb
from typing import Callable

import numpy as np
import polars as pl

from pipeline.config import ValidationConfig
from pipeline.splits import build_train_segments, make_groups
from analysis.metrics import (
    annualized_return, block_bootstrap_sharpe, deflated_sharpe_ratio,
    max_drawdown, probabilistic_sharpe_ratio, sharpe_ratio,
)
from analysis.report import Severity, ValidationReport


BacktestRunner = Callable[
    [list[pl.DataFrame], pl.DataFrame],
    tuple[pl.Series, pl.Series],
]


@dataclass
class CPCVConfig:
    n_groups: int = 6        # N
    n_test_groups: int = 2   # k  (k=2 da el mayor número de trayectorias para N dado)


def n_paths(n: int, k: int) -> int:
    """φ = C(N−1, k−1)  (identidad combinatoria: = C(N,k)·k/N)"""
    return comb(n - 1, k - 1)


class CPCVEngine:
    """
    Para cada una de las C(N,k) combinaciones de k grupos de test:
      1. Construye los segmentos de entrenamiento con purging y embargo.
      2. Llama al runner una vez por grupo de test de la combinación.
         (Grupos separados evitan continuidad artificial entre períodos no contiguos.)
      3. Reconstruye φ trayectorias OOS completas asignando resultados round-robin
         por grupo (la p-ésima combinación que incluye al grupo i va al path p).
    """

    def __init__(self, val_config: ValidationConfig, cpcv_config: CPCVConfig) -> None:
        self._cfg  = val_config
        self._cpcv = cpcv_config
        self.paths_returns: list[pl.Series] = []

    def run(self, data: pl.DataFrame, runner: BacktestRunner) -> ValidationReport:
        report = ValidationReport()
        report.date_range    = (
            str(data["timestamp"].min())[:10],
            str(data["timestamp"].max())[:10],
        )
        report.bars_per_year = self._cfg.bars_per_year
        n, k = self._cpcv.n_groups, self._cpcv.n_test_groups

        if k >= n:
            report.add(Severity.ERROR, "cpcv_k_vs_n",
                       f"k={k} debe ser menor que N={n}.")
            return report

        phi = n_paths(n, k)
        if phi < 2:
            report.add(Severity.ERROR, "cpcv_min_paths",
                       f"C({n},{k}) produce phi={phi} trayectorias -- se necesitan al menos 2.")
            return report

        groups = self._make_groups(data)
        min_group_size = min(len(g) for g in groups)
        if min_group_size < 10:
            report.add(Severity.ERROR, "cpcv_group_size",
                       f"Grupo más pequeño tiene {min_group_size} barras. "
                       "Reducir N o aumentar el dataset.")
            return report

        all_combos = list(_combinations(range(n), k))
        total_runner_calls = len(all_combos) * k
        if total_runner_calls > 500:
            report.add(Severity.WARNING, "cpcv_cost",
                       f"C({n},{k})×k = {total_runner_calls} llamadas al runner. "
                       "Considerar reducir N o k.")

        # group_results[group_idx][combo] = returns
        group_results: dict[int, dict[tuple, pl.Series]] = {
            i: {} for i in range(n)
        }
        for combo in all_combos:
            test_set   = set(combo)
            train_segs = self._get_train_segments(groups, test_set)
            if not train_segs:
                report.add(Severity.WARNING, f"cpcv_empty_train_{combo}",
                           f"Combinación {combo}: segmentos de train vacíos tras purging/embargo.")
                continue
            for group_idx in combo:
                rets, _ = runner(train_segs, groups[group_idx])
                group_results[group_idx][combo] = rets

        paths_returns = self._reconstruct_paths(
            group_results, all_combos, n, phi
        )
        self.paths_returns = paths_returns
        report.oos_returns = [r.to_numpy() for r in paths_returns]

        if not paths_returns:
            report.add(Severity.ERROR, "cpcv_no_paths", "No se pudieron reconstruir trayectorias.")
            return report

        # ── métricas por trayectoria ─────────────────────────────────────────
        bpy     = self._cfg.bars_per_year
        sharpes = [sharpe_ratio(r, bpy) for r in paths_returns]
        report.metrics["phi"]               = phi
        report.metrics["sharpes_per_path"]  = sharpes
        report.metrics["n_combos"]          = len(all_combos)

        report.add(Severity.INFO, "phi",
                   f"phi={phi} trayectorias OOS  |  C({n},{k})={len(all_combos)} combinaciones")
        report.add(Severity.INFO, "sharpe_mean",
                   f"Sharpe medio (phi paths): {np.mean(sharpes):.3f}")
        report.add(Severity.INFO, "sharpe_std",
                   f"Sharpe std  (phi paths): {np.std(sharpes):.3f}")
        report.add(Severity.INFO, "sharpe_p5",
                   f"Sharpe p5   (phi paths): {float(np.percentile(sharpes, 5)):.3f}")

        pct_pos = sum(s > 0 for s in sharpes) / len(sharpes)
        sev = Severity.WARNING if pct_pos < 0.60 else Severity.INFO
        report.add(sev, "pct_positive_paths",
                   f"Trayectorias con Sharpe > 0: {pct_pos:.0%}")

        # ── métricas sobre el promedio de trayectorias ───────────────────────
        min_len     = min(len(r) for r in paths_returns)
        avg_returns = pl.Series("avg_returns", np.mean(
            np.stack([r.to_numpy()[:min_len] for r in paths_returns]), axis=0
        ))

        sr_avg    = sharpe_ratio(avg_returns, bpy)
        psr_avg   = probabilistic_sharpe_ratio(avg_returns)
        mdd_avg   = max_drawdown(avg_returns)
        ann_r_avg = annualized_return(avg_returns, bpy)

        report.metrics.update({
            "sharpe_avg_path":       sr_avg,
            "psr_avg_path":          psr_avg,
            "max_drawdown":          mdd_avg,
            "annualized_return_avg": ann_r_avg,
        })
        report.add(Severity.INFO, "annualized_return_avg",
                   f"Retorno anualizado (avg path): {ann_r_avg:.2%}")
        report.add(Severity.INFO, "sharpe_avg_path", f"Sharpe path-promedio: {sr_avg:.3f}")
        sev = Severity.WARNING if psr_avg < 0.70 else Severity.INFO
        report.add(sev, "psr_avg_path", f"PSR path-promedio: {psr_avg:.1%}")
        report.add(Severity.INFO, "max_drawdown", f"Max drawdown (avg path): {mdd_avg:.2%}")

        if self._cfg.n_trials > 1:
            dsr = deflated_sharpe_ratio(avg_returns, self._cfg.n_trials)
            report.metrics["dsr"] = dsr
            sev = Severity.WARNING if (np.isnan(dsr) or dsr < 0.95) else Severity.INFO
            label = f"{dsr:.1%}" if not np.isnan(dsr) else "nan"
            report.add(sev, "dsr", f"DSR ({self._cfg.n_trials} trials): {label}")

        if self._cfg.block_bootstrap_reps > 0:
            block_len = self._cfg.alpha_halflife_bars or 20
            boot = block_bootstrap_sharpe(avg_returns, self._cfg.block_bootstrap_reps, block_len)
            report.metrics["bootstrap"] = boot
            sev = Severity.WARNING if (np.isnan(boot["p5"]) or boot["p5"] < 0) else Severity.INFO
            report.add(sev, "bootstrap_p5", f"Bootstrap p5 Sharpe: {boot['p5']:.3f}")

        return report

    # ── grupos ───────────────────────────────────────────────────────────────

    def _make_groups(self, data: pl.DataFrame) -> list[pl.DataFrame]:
        return make_groups(data, self._cpcv.n_groups)

    # ── purging y embargo por combinación ────────────────────────────────────

    def _get_train_segments(
        self,
        groups: list[pl.DataFrame],
        test_set: set[int],
    ) -> list[pl.DataFrame]:
        return build_train_segments(
            groups,
            test_set,
            self._cfg.label_horizon,
            self._cfg.embargo_pct,
            self._cfg.embargo_bars,
        )

    # ── reconstrucción de trayectorias ───────────────────────────────────────

    def _reconstruct_paths(
        self,
        group_results: dict[int, dict[tuple, pl.Series]],
        all_combos: list[tuple],
        n: int,
        phi: int,
    ) -> list[pl.Series]:
        paths: list[list[pl.Series]] = [[] for _ in range(phi)]

        for group_idx in range(n):
            combos_with_group = sorted(c for c in all_combos if group_idx in c)
            for p, combo in enumerate(combos_with_group):
                if p >= phi:
                    break
                if combo not in group_results.get(group_idx, {}):
                    continue
                paths[p].append(group_results[group_idx][combo])

        valid = [i for i, segs in enumerate(paths) if segs and all(len(s) > 0 for s in segs)]
        return [pl.concat(paths[i]) for i in valid]
