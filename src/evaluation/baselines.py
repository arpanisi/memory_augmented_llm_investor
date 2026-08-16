"""
Baselines evaluation module (Buy-and-Hold & Equal-Weight).
Uses close(t)-to-close(t+1) return attribution convention identical to full model.
"""
import pandas as pd
import numpy as np

def run_buy_and_hold_baseline(price_pivot: pd.DataFrame) -> pd.Series:
    """
    Buy-and-Hold: Equal-weight the universe once at the start of the evaluation window (w_i = 1/N).
    Never rebalance.
    Returns daily close(t)-to-close(t+1) portfolio return series.
    """
    daily_returns = price_pivot.pct_change()
    dates = daily_returns.index[:-1]
    
    # Portfolio value starting at 1.0 at start of window
    N = len(price_pivot.columns)
    initial_shares = (1.0 / N) / price_pivot.iloc[0]

    portfolio_values = []
    for d in price_pivot.index:
        val = float(np.sum(initial_shares * price_pivot.loc[d]))
        portfolio_values.append(val)

    pv_series = pd.Series(portfolio_values, index=price_pivot.index)
    # close(t) to close(t+1) return
    rets = (pv_series.shift(-1) - pv_series) / pv_series
    return rets.iloc[:-1]

def run_equal_weight_baseline(price_pivot: pd.DataFrame) -> pd.Series:
    """
    Equal-Weight: w_i = 1/N for every asset, rebalanced daily.
    Returns daily close(t)-to-close(t+1) portfolio return series.
    """
    daily_returns = price_pivot.pct_change()
    # Daily return earned close(t)-to-close(t+1) is daily_returns.shift(-1)
    ew_daily_rets = daily_returns.mean(axis=1).shift(-1).iloc[:-1]
    return ew_daily_rets
