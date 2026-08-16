from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from system.util import (
    rebase, drawdown, equity,
    sharpe_ratio, sortino_ratio,
    yearly_sharpe, yearly_sortino,
    rolling_alpha_beta,
)

plt.style.use('classic')
plt.rcParams['lines.linewidth'] = 1.5


def plot_equity_drawdown(port_ret, bench_ret, start=None):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    p_ec = rebase(equity(port_ret), start)
    b_ec = rebase(equity(bench_ret), start)

    p_dd = -drawdown(p_ec) * 100
    b_dd = -drawdown(b_ec) * 100

    axes[0].set_title('Equity Curve')
    axes[0].plot(p_ec, label='Portfolio', alpha=0.85)
    axes[0].plot(b_ec, label='Benchmark', alpha=0.85)
    axes[0].legend(loc='upper left')
    axes[0].set_ylabel('Factor')

    axes[1].set_title('Drawdown')
    axes[1].plot(p_dd)
    axes[1].plot(b_dd)
    axes[1].set_ylabel('Percent')

    for a in axes:
        a.grid()

    plt.tight_layout()
    return fig


def plot_shape(port_ret, bench_ret, start=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    p_ret = port_ret if start is None else port_ret[start:]
    b_ret = bench_ret if start is None else bench_ret[start:]

    shape_df = pd.DataFrame({
        'Portfolio': p_ret.describe() * 100,
        'Benchmark': b_ret.describe() * 100,
    })

    shape_df.loc[['25%', '50%', '75%', 'mean', 'std']].plot(
        kind='barh', ax=axes[0], alpha=0.8, title='Shape'
    )

    kurt_df = pd.DataFrame({
        'Portfolio': p_ret.kurt(),
        'Benchmark': b_ret.kurt(),
    }, index=['Kurtosis'])

    kurt_df.plot(kind='bar', ax=axes[1], alpha=0.8, title='Kurtosis', ylabel='%')

    for ax in axes:
        ax.grid()

    plt.tight_layout()
    return fig


def plot_alpha_beta(port_ret, bench_ret, start=None, window=60):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    p_ret = port_ret if start is None else port_ret[start:]
    b_ret = bench_ret if start is None else bench_ret[start:]

    rolling_alpha, rolling_beta = rolling_alpha_beta(p_ret, b_ret, window=window)

    rolling_beta.plot(title=f'{window}-Day Rolling Beta', ax=axes[0])
    axes[0].axhline(1, linestyle='--')

    rolling_alpha.plot(title=f'{window}-Day Rolling Alpha', ax=axes[1])
    axes[1].axhline(0, linestyle='--')

    for ax in axes:
        ax.grid()

    plt.tight_layout()
    return fig


def plot_benchmark_ratios(port_ret, bench_ret, start=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    p_ret = port_ret if start is None else port_ret[start:]
    b_ret = bench_ret if start is None else bench_ret[start:]

    p_sharpe = sharpe_ratio(p_ret)
    b_sharpe = sharpe_ratio(b_ret)

    sharpe_df = pd.DataFrame({
        'Portfolio': yearly_sharpe(p_ret),
        'Benchmark': yearly_sharpe(b_ret),
    })
    sharpe_df.plot(
        kind='bar', ax=axes[0],
        title=f'Yearly Sharpe Ratio | All-Time: Portfolio {p_sharpe:.2f} | Benchmark {b_sharpe:.2f}'
    )

    p_sortino = sortino_ratio(p_ret)
    b_sortino = sortino_ratio(b_ret)

    sortino_df = pd.DataFrame({
        'Portfolio': yearly_sortino(p_ret),
        'Benchmark': yearly_sortino(b_ret),
    })
    sortino_df.plot(
        kind='bar', ax=axes[1],
        title=f'Yearly Sortino Ratio | All-Time: Portfolio {p_sortino:.2f} | Benchmark {b_sortino:.2f}'
    )

    for ax in axes:
        ax.grid(axis='y')
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_xlabel('Year')

    plt.tight_layout()
    return fig


def export_report_pdf(port_ret, bench_ret, start=None, path='backtest_report.pdf'):
    with PdfPages(path) as pdf:
        for plot_fn in (plot_equity_drawdown, plot_shape, plot_alpha_beta, plot_benchmark_ratios):
            fig = plot_fn(port_ret, bench_ret, start)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'Report saved to: {path}')