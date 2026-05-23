from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ReportEntry:
    severity: Severity
    check: str
    message: str
    value: Any = None


@dataclass
class ValidationReport:
    entries: list[ReportEntry] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Poblados automaticamente por CPCVEngine al finalizar run()
    oos_returns: Any = field(default=None, repr=False, compare=False)
    date_range: tuple[str, str] | None = field(default=None, repr=False, compare=False)
    bars_per_year: int = field(default=252, repr=False, compare=False)

    def add(self, severity: Severity, check: str, message: str, value: Any = None) -> None:
        self.entries.append(ReportEntry(severity, check, message, value))

    def has_errors(self) -> bool:
        return any(e.severity == Severity.ERROR for e in self.entries)

    def plot_vs_spy(
        self,
        output: str = "equity_vs_benchmark.png",
        strategy_label: str = "Estrategia",
        benchmark_ticker: str = "SPY",
        title: str | None = None,
    ) -> None:
        """
        Genera el grafico de equity de la estrategia vs. benchmark.
        Requiere haber corrido CPCVEngine.run() antes.
        """
        if self.oos_returns is None:
            raise RuntimeError("Sin retornos OOS. Correr CPCVEngine.run() primero.")
        if self.date_range is None:
            raise RuntimeError("Sin rango de fechas. El DataFrame debe tener columna 'timestamp'.")

        from analysis.plots import plot_equity_vs_benchmark

        plot_equity_vs_benchmark(
            self.oos_returns,
            start_date=self.date_range[0],
            end_date=self.date_range[1],
            strategy_label=strategy_label,
            benchmark_ticker=benchmark_ticker,
            output=output,
            title=title,
            bars_per_year=self.bars_per_year,
        )

    def summary(self) -> str:
        m   = self.metrics
        W   = 60
        sep = "═" * W
        thin = "─" * W

        def pct(v):   return f"{v:+.2%}"  if v is not None else "n/a"
        def sr(v):    return f"{v:.3f}"   if v is not None else "n/a"
        def prob(v):  return f"{v:.1%}"   if v is not None else "n/a"
        def row(label, value):
            return f"  {label:<32}{value}"

        lines = []

        # ── encabezado ───────────────────────────────────────────────────────
        is_cpcv  = "sharpes_per_path" in m

        if is_cpcv:
            title = f"CPCV  ·  C={m.get('n_combos','?')} combinaciones  ·  φ={m.get('phi','?')} trayectorias"
        else:
            title = "VALIDATION REPORT"

        date_str = (f"  {self.date_range[0]} → {self.date_range[1]}"
                    if self.date_range else "")

        lines += [sep, f"  {title}{date_str}", sep, ""]

        # ── métricas principales ─────────────────────────────────────────────
        if is_cpcv:
            lines += [
                row("Retorno anualizado",  pct(m.get("annualized_return_avg"))),
                row("Sharpe (avg path)",   sr(m.get("sharpe_avg_path"))),
                row("Max Drawdown",        pct(m.get("max_drawdown"))),
                row("PSR (avg path)",      prob(m.get("psr_avg_path"))),
            ]
            if "dsr" in m:
                lines.append(row("DSR", prob(m["dsr"])))
            if "bootstrap" in m:
                lines.append(row("Bootstrap Sharpe p5", sr(m["bootstrap"].get("p5"))))

            lines.append("")

            path_sharpes = m.get("sharpes_per_path", [])
            if path_sharpes:
                phi = len(path_sharpes)
                header = "  " + "".join(f"   P{i+1} " for i in range(phi))
                values = "  " + "".join(
                    f" {s:+.2f}" if s >= 0 else f" {s:.2f}" for s in path_sharpes
                )
                sr_mean = float(np.mean(path_sharpes))
                sr_std  = float(np.std(path_sharpes))
                sr_p5   = float(np.percentile(path_sharpes, 5))
                n_pos   = sum(1 for s in path_sharpes if s > 0)
                lines += [
                    "  Sharpe por trayectoria:",
                    header,
                    values,
                    f"  media {sr_mean:.3f}  ·  std {sr_std:.3f}  ·  p5 {sr_p5:.3f}",
                    f"  {n_pos}/{phi} paths positivos ({n_pos/phi:.0%})",
                ]

        else:
            for e in self.entries:
                if e.severity == Severity.INFO:
                    lines.append(f"  {e.message}")

        # ── alertas ──────────────────────────────────────────────────────────
        warnings = [e for e in self.entries if e.severity == Severity.WARNING]
        errors   = [e for e in self.entries if e.severity == Severity.ERROR]

        if errors or warnings:
            lines += ["", thin]
            for e in errors:
                lines.append(f"  ✗  {e.message}")
            for e in warnings:
                lines.append(f"  ⚠  {e.message}")

        lines.append(sep)
        return "\n".join(lines)
