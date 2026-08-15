
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use('classic')
plt.rcParams['lines.linewidth'] = 1.5

from system.util import rebase, drawdown, equity
from system.portfolio import (
    get_first_trade_date
    , get_portfolio_returns
    , get_spy_returns
    , get_portfolio_returns_levered
    , get_spy_equity
)


def plot_portfolio_equity(cfg, state, **kwargs):
    fig, axes = plt.subplots(2, 1, figsize=(16,10))

    start = get_first_trade_date(cfg, state)

    if getattr(cfg, 'leverage', False):
        p_ret = get_portfolio_returns_levered(cfg, state, **kwargs)
    else:
        p_ret = get_portfolio_returns(cfg, state, **kwargs)

    p_ec = rebase(equity(p_ret), start)
    b_ec = rebase(get_spy_equity(cfg, state, **kwargs), start)

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

    return fig

def plot_shape(cfg, state):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    start = get_first_trade_date(cfg, state)
    p_ret = get_portfolio_returns(cfg, state)[start:]
    sp_ret = get_spy_returns(cfg, state)[start:]

    shape_df = pd.DataFrame({
        'Portfolio': p_ret.describe() * 100,
        'SPY': sp_ret.describe() * 100
    })

    shape_df.loc[['25%', '50%', '75%', 'mean', 'std']].plot(
        kind='barh'
        , ax=axes[0]
        , alpha=0.8
        , title='Shape'
    )

    kurt_df = pd.DataFrame({
        'Portfolio': p_ret.kurt(),
        'SPY': sp_ret.kurt()
    }, index=['Kurtosis'])

    kurt_df.plot(
        kind='bar'
        , ax=axes[1]
        , alpha=0.8
        , title='Kurtosis'
        , ylabel='%'
    )

    for ax in axes:
        ax.grid()

    return fig

# TODO alpha beta baked into visualizations these should be util functions
def plot_alpha_beta(cfg, state):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    start = get_first_trade_date(cfg, state)

    p_ret = get_portfolio_returns(cfg, state)[start:]
    sp_ret = get_spy_returns(cfg, state)[start:]

    window = 60

    rolling_beta = (
        p_ret.rolling(window).cov(sp_ret)
        / sp_ret.rolling(window).var()
    )

    rolling_alpha = (
        p_ret.rolling(window).mean()
        - rolling_beta * sp_ret.rolling(window).mean()
    ) * 252

    rolling_beta.plot(title='60-Day Rolling Beta', ax=axes[0])
    axes[0].axhline(1, linestyle='--')

    rolling_alpha.plot(title='60-Day Rolling Alpha', ax=axes[1])
    axes[1].axhline(0, linestyle='--')

    for ax in axes:
        ax.grid()

    plt.tight_layout()
    return fig

# TODO same issue as previous todo
def plot_benchmark_ratios(cfg, state):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    start = get_first_trade_date(cfg, state)

    p_ret = get_portfolio_returns(cfg, state)[start:]
    sp_ret = get_spy_returns(cfg, state)[start:]

    # -------------------------
    # All-Time Sharpe
    # -------------------------

    p_sharpe = (p_ret.mean() / p_ret.std()) * np.sqrt(252)
    sp_sharpe = (sp_ret.mean() / sp_ret.std()) * np.sqrt(252)

    # -------------------------
    # Yearly Sharpe
    # -------------------------

    p_yearly_sharpe = (
        p_ret.groupby(p_ret.index.year).mean()
        / p_ret.groupby(p_ret.index.year).std()
    ) * np.sqrt(252)

    sp_yearly_sharpe = (
        sp_ret.groupby(sp_ret.index.year).mean()
        / sp_ret.groupby(sp_ret.index.year).std()
    ) * np.sqrt(252)

    sharpe_df = pd.DataFrame({
        'Portfolio': p_yearly_sharpe,
        'SPY': sp_yearly_sharpe
    })

    sharpe_df.plot(
        kind='bar',
        ax=axes[0],
        title=f'Yearly Sharpe Ratio | All-Time: Portfolio {p_sharpe:.2f} | SPY {sp_sharpe:.2f}'
    )

    # -------------------------
    # All-Time Sortino
    # -------------------------

    p_downside = p_ret.where(p_ret < 0, 0)
    sp_downside = sp_ret.where(sp_ret < 0, 0)

    p_sortino = (
        p_ret.mean()
        / np.sqrt(np.mean(p_downside ** 2))
    ) * np.sqrt(252)

    sp_sortino = (
        sp_ret.mean()
        / np.sqrt(np.mean(sp_downside ** 2))
    ) * np.sqrt(252)

    # -------------------------
    # Yearly Sortino
    # -------------------------

    p_yearly_sortino = (
        p_ret.groupby(p_ret.index.year).mean()
        / p_downside.groupby(p_downside.index.year).apply(
            lambda x: np.sqrt(np.mean(x ** 2))
        )
    ) * np.sqrt(252)

    sp_yearly_sortino = (
        sp_ret.groupby(sp_ret.index.year).mean()
        / sp_downside.groupby(sp_downside.index.year).apply(
            lambda x: np.sqrt(np.mean(x ** 2))
        )
    ) * np.sqrt(252)

    sortino_df = pd.DataFrame({
        'Portfolio': p_yearly_sortino,
        'SPY': sp_yearly_sortino
    })

    sortino_df.plot(
        kind='bar',
        ax=axes[1],
        title=f'Yearly Sortino Ratio | All-Time: Portfolio {p_sortino:.2f} | SPY {sp_sortino:.2f}'
    )

    for ax in axes:
        ax.grid(axis='y')
        ax.axhline(0, linestyle='--', linewidth=1)
        ax.set_xlabel('Year')

    plt.tight_layout()
    return fig

def export_report_pdf(cfg, state, path='backtest_report.pdf'):
    with PdfPages(path) as pdf:

        # Page 1: Equity Curve + Drawdown
        fig = plot_portfolio_equity(cfg, state)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: Return Distribution + Kurtosis
        fig = plot_shape(cfg, state)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 3: Rolling Alpha + Beta
        fig = plot_alpha_beta(cfg, state)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 4: Yearly Sharpe + Sortino
        fig = plot_benchmark_ratios(cfg, state)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f'Report saved to: {path}')