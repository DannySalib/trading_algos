import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def trading_dashboard(strategy_returns: pd.Series, benchmark_returns: pd.Series = None, title: str = "Strategy"):
    """
    Visualize any trading strategy given its daily return series.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily log or simple returns from your strategy, DatetimeIndex.
    benchmark_returns : pd.Series, optional
        Buy-and-hold or any benchmark returns to compare against.
    title : str
        Chart title.

    Returns
    -------
    dict  —  sharpe, profit_factor, total_return, win_rate

    Example
    -------
    # Define your strategy however you want, then just pass the returns:
    df['signal'] = np.where(fast_ma > slow_ma, 1, 0)
    df['strat_returns'] = df['signal'] * np.log(df['Close']).diff().shift(-1)

    stats = trading_dashboard(df['strat_returns'].dropna(), title="MA Crossover")
    """
    r = strategy_returns.dropna()

    # --- Metrics ---
    sharpe        = (r.mean() / r.std()) * np.sqrt(252)
    profit_factor = r[r > 0].sum() / r[r < 0].abs().sum() if r[r < 0].abs().sum() != 0 else np.inf
    total_return  = np.exp(r.sum()) - 1
    win_rate      = (r > 0).sum() / (r != 0).sum()

    strat_equity  = np.exp(r.cumsum())

    # --- Plot ---
    fig = plt.figure(figsize=(13, 8), facecolor='#0f0f1a')
    fig.suptitle(title, fontsize=14, fontweight='bold', color='white', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(strat_equity.index, strat_equity, color='#4CE882', linewidth=1.5, label='Strategy')
    if benchmark_returns is not None:
        bm = benchmark_returns.reindex(r.index).fillna(0)
        ax1.plot(strat_equity.index, np.exp(bm.cumsum()), color='#aaa', linewidth=1.1, label='Benchmark', linestyle='--')
    ax1.axhline(1.0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
    ax1.set_facecolor('#1a1a2e'); ax1.grid(True, alpha=0.2)
    ax1.set_ylabel('Growth of $1', color='white'); ax1.set_title('Cumulative Equity', color='white')
    ax1.legend(fontsize=9); ax1.tick_params(colors='white')

    # 2. Rolling 60-day Sharpe
    ax2 = fig.add_subplot(gs[1, 0])
    rolling_sharpe = (r.rolling(60).mean() / r.rolling(60).std()) * np.sqrt(252)
    ax2.plot(rolling_sharpe.index, rolling_sharpe, color='#C47FE8', linewidth=1.1)
    ax2.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
    ax2.set_facecolor('#1a1a2e'); ax2.grid(True, alpha=0.2)
    ax2.set_title('Rolling 60-Day Sharpe', color='white'); ax2.tick_params(colors='white')

    # 3. Stats table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    rows = [
        ['Annualized Sharpe', f'{sharpe:.3f}'],
        ['Profit Factor',     f'{profit_factor:.3f}'],
        ['Total Return',      f'{total_return:.1%}'],
        ['Win Rate',          f'{win_rate:.1%}'],
    ]
    tbl = ax3.table(cellText=rows, colLabels=['Metric', 'Value'], loc='center', cellLoc='left')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#444')
        cell.set_facecolor('#2a2a3e' if row == 0 else '#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold' if row == 0 else 'normal')
    ax3.set_title('Performance Summary', color='white', pad=12)

    plt.show()
    return {'sharpe': round(sharpe, 4), 'profit_factor': round(profit_factor, 4),
            'total_return': round(total_return, 4), 'win_rate': round(win_rate, 4)}