"""CLI de RMT Stat-Arb. Uso: python -m rmt_stat_arb <comando>"""
import argparse
import sys
from pathlib import Path

# rmt_stat_arb/ en sys.path para imports internos
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_backtest(args):
    from scripts.run_validation_rmt import main
    main()


def cmd_paper(args):
    from scripts.run_paper import run_paper_trading
    run_paper_trading(force=args.force)


def cmd_status(args):
    from monitoring.status import show_status
    show_status()


def main():
    parser = argparse.ArgumentParser(
        prog="rmt",
        description="RMT Stat-Arb — sistema de trading cuantitativo (backtest + paper trading + monitoreo)",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<comando>")

    sub.add_parser("backtest", help="Corre la validación CPCV completa con stress testing")

    p_paper = sub.add_parser("paper", help="Ejecuta un rebalanceo de paper trading contra IBKR")
    p_paper.add_argument("--force", action="store_true",
                         help="Saltea check de idempotencia (correr más de una vez por día)")

    sub.add_parser("status", help="Muestra el estado actual del portfolio sin operar")

    args = parser.parse_args()
    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "paper":
        cmd_paper(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
