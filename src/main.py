from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import typing
import os
from pathlib import Path

from system.config import Config
from system.state import State
from system.signal.portfolio import CsmGePortfolioSignal
from system.accessors import get_tickers_returns, get_tbill_returns, get_global_market_returns, get_spy_returns
from system.util import portfolio_returns, first_trade_date, sharpe_ratio, sortino_ratio, max_drawdown
from system.visualize import export_report_pdf

BACKTEST_REPORT_PATH = Path(__file__).parent / 'reports'
BACKTEST_REPORT_PATH.mkdir(exist_ok=True, parents=True)

def _str_to_bool(v: str) -> bool:
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def _str_to_date(v: str) -> dt.date:
    return dt.date.fromisoformat(v)


_TYPE_COERCERS = {
    bool: _str_to_bool,
    int: int,
    float: float,
    str: str,
    dt.date: _str_to_date,
}


def build_config_from_args() -> Config:
    hints = typing.get_type_hints(Config)
    parser = argparse.ArgumentParser()

    for f in dataclasses.fields(Config):
        field_type = hints.get(f.name)
        coercer = _TYPE_COERCERS.get(field_type)
        if coercer is None:
            continue  # complex/nested field (e.g. param, universe) -- not flag-able as a flat string
        parser.add_argument(f'--{f.name}', type=coercer, default=None)

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return Config(**overrides)


def run_backtest(cfg: Config, state: State):
    signal_df = CsmGePortfolioSignal.signal(cfg, state)

    tickers_ret = get_tickers_returns(cfg, state)
    tbill_ret = get_tbill_returns(cfg, state)
    global_ret = get_global_market_returns(cfg, state)
    bench_ret = get_spy_returns(cfg, state)

    port_ret = portfolio_returns(signal_df, tickers_ret, tbill_ret, global_ret)
    start = first_trade_date(signal_df)

    port_ret = port_ret.loc[start:]
    bench_ret = bench_ret.reindex(port_ret.index)

    return port_ret, bench_ret


def print_summary(port_ret, bench_ret):
    print(f"{'':12s}{'Portfolio':>12s}{'Benchmark':>12s}")
    print(f"{'Sharpe':12s}{sharpe_ratio(port_ret):12.2f}{sharpe_ratio(bench_ret):12.2f}")
    print(f"{'Sortino':12s}{sortino_ratio(port_ret):12.2f}{sortino_ratio(bench_ret):12.2f}")
    print(f"{'Max DD':12s}{max_drawdown(port_ret)*100:11.2f}%{max_drawdown(bench_ret)*100:11.2f}%")


def main():
    cfg = build_config_from_args()
    state = State()
    port_ret, bench_ret = run_backtest(cfg, state)
    print_summary(port_ret, bench_ret)
    export_report_pdf(port_ret, bench_ret, path=BACKTEST_REPORT_PATH / 'backtest_report.pdf')
    os.startfile(BACKTEST_REPORT_PATH / 'backtest_report.pdf')

if __name__ == '__main__':
    main()