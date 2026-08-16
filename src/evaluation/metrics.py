"""
Performance metrics evaluation module.
Calculates Annualized Sharpe (sqrt(252)), Cumulative Return, and Maximum Drawdown (MDD).
"""
import numpy as np
import pandas as pd

def calculate_annualized_sharpe(daily_returns: pd.Series) -> float:
    """
    SR = sqrt(252) * mean(R_p) / std(R_p)
    """
    rets = daily_returns.dropna().values
    if len(rets) == 0:
        return 0.0
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    if std_ret == 0.0:
        return 0.0
    return float(np.sqrt(252) * (mean_ret / std_ret))

def calculate_cumulative_return(daily_returns: pd.Series) -> float:
    """
    Cumulative return = prod(1 + R_p) - 1
    """
    rets = daily_returns.dropna().values
    if len(rets) == 0:
        return 0.0
    equity_curve = np.cumprod(1.0 + rets)
    return float(equity_curve[-1] - 1.0)

def calculate_max_drawdown(daily_returns: pd.Series) -> float:
    """
    MDD = min_t (Trough_t - Peak_t) / Peak_t
    """
    rets = daily_returns.dropna().values
    if len(rets) == 0:
        return 0.0
    cum_returns = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    return float(np.min(drawdowns))

def compute_evaluation_summary(daily_returns: pd.Series, model_name: str = "Model") -> dict:
    """
    Returns complete metric dictionary for a return series.
    """
    return {
        "model_name": model_name,
        "annualized_sharpe": calculate_annualized_sharpe(daily_returns),
        "cumulative_return": calculate_cumulative_return(daily_returns),
        "max_drawdown": calculate_max_drawdown(daily_returns),
        "trading_days": len(daily_returns.dropna())
    }
