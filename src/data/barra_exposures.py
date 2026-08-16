"""
Barra-style exposure lookup for traded assets (Step 5 in the locked plan).

This module holds the decision-time lookup side of the Barra pipeline: reading the estimation
universe's already-computed daily cross-sectional style stats (from `barra_estimation`) and
building the (N, 14) factor-exposure matrix used by the portfolio optimizer and the realized
factor-exposure report. It also contains the raw style-row builder (`_compute_daily_stock_row`)
used to assemble the estimation universe's daily cross-sections, plus the small shared helpers
(SIC-division mapping, guarded z-scoring) used by both pipeline steps.
"""
import numpy as np
import pandas as pd
from config.settings import SIC_DIVISIONS, CRYPTO_BOOK, ETF_BOOK
from src.data.edgar import get_point_in_time_fundamentals, get_sic_code


def get_sic_division(sic_code: str) -> str:
    """
    Maps 2-digit SIC code to one of 10 static SIC Divisions (A through J).
    """
    try:
        sic_num = int(str(sic_code)[:2])
    except (ValueError, TypeError):
        return "D"  # Fallback to Manufacturing if unavailable

    for div, sic_range in SIC_DIVISIONS.items():
        if sic_num in sic_range:
            return div
    return "D"


def _z_score(x: float, mean: float, std: float) -> float:
    """z = (x - mean) / std, guarded against zero/NaN std (returns neutral 0.0)."""
    if std is None or pd.isna(std) or std <= 0:
        return 0.0
    if mean is None or pd.isna(mean) or x is None or pd.isna(x):
        return 0.0
    return float((x - mean) / std)


def _compute_daily_stock_row(
    t: str, r_it: float, p_it: float, rets_252, vol_60, sic_map: dict, rf_val: float,
    decision_date: str, pit_fundamentals: pd.DataFrame | None = None,
) -> dict | None:
    """
    Builds the raw style row for estimation-universe stock `t` on a day, or None if any required
    point-in-time fundamental (shares outstanding / stockholders' equity) is unavailable.

    Point-in-time semantics: when `pit_fundamentals` is provided (a per-ticker panel built by
    edgar.build_pit_fundamentals_panel, indexed by decision date) its row for the day carries the
    most recent `filed` <= decision_date value. Otherwise get_point_in_time_fundamentals is used.
    In either path a missing value excludes the stock from the cross-sectional regression rather
    than substituting a fabricated number.
    """
    if pit_fundamentals is not None:
        row = pit_fundamentals.loc[decision_date]
        shares = row["shares_outstanding"]
        equity = row["stockholders_equity"]
    else:
        fund = get_point_in_time_fundamentals(t, decision_date=decision_date, closing_price=p_it)
        shares = fund.get("shares_outstanding")
        equity = fund.get("stockholders_equity")
    # A NaN from the panel (no filing filed on/before the day) is missing, not valid.
    if shares is None or equity is None or pd.isna(shares) or pd.isna(equity) or shares <= 0:
        return None

    mcap = p_it * shares
    size_raw = np.log(max(mcap, 1e4))
    value_raw = equity / max(mcap, 1e4)
    mom_raw = rets_252.get(t, 0.0)
    vol_raw = vol_60.get(t, 0.15)

    return {
        "ticker": t,
        "excess_return": r_it - rf_val,
        "mcap": mcap,
        "division": sic_map.get(t, "D"),
        "size_raw": size_raw,
        "value_raw": value_raw,
        "mom_raw": mom_raw if not pd.isna(mom_raw) else 0.0,
        "vol_raw": vol_raw if not pd.isna(vol_raw) else 0.15,
    }


def compute_asset_factor_exposures(
    price_pivot: pd.DataFrame,
    traded_equities: list[str],
    estimation_tickers: list[str],
    decision_date: str,
    style_stats: dict,
) -> np.ndarray | None:
    """
    Computes the (N, 14) factor exposure matrix for traded_equities on decision_date:
      10 industry-division dummies (1/0 from the filer's SIC division) + 4 style z-scores.

    Style z-scores (Size, Value, Momentum, Volatility) are computed the same way as for the
    estimation universe and normalized against that day's estimation-universe cross-sectional
    mean/std (via `style_stats`), NOT the traded universe's own statistics. Crypto/ETF assets get
    structural 0 exposure to all 14 factors. A traded equity whose point-in-time shares/equity value
    is unavailable gets a neutral 0.0 style exposure (never an invented number).

    Returns None if the estimation-universe stats for the day are unavailable (caller falls back).
    """
    date_str = str(pd.Timestamp(decision_date).strftime("%Y-%m-%d"))
    if date_str not in style_stats:
        return None
    if date_str not in price_pivot.index:
        return None

    idx = price_pivot.index.get_loc(date_str)
    if idx < 252:
        return None

    rets_252 = (price_pivot.iloc[idx - 21] / price_pivot.iloc[idx - 252]) - 1.0
    daily_returns = price_pivot.pct_change()
    vol_60 = daily_returns.iloc[idx - 60:idx].std() * np.sqrt(252)

    division_order = list(SIC_DIVISIONS.keys())
    N = len(traded_equities)
    X = np.zeros((N, 14))

    for i, t in enumerate(traded_equities):
        if t in CRYPTO_BOOK or t in ETF_BOOK:
            # Structural zero exposure: no SEC filings.
            continue

        p_t = price_pivot.loc[date_str, t] if t in price_pivot.columns else np.nan
        if pd.isna(p_t) or p_t <= 0:
            continue

        fund = get_point_in_time_fundamentals(t, decision_date=date_str, closing_price=float(p_t))
        shares = fund.get("shares_outstanding")
        equity = fund.get("stockholders_equity")

        # Industry dummies from the filer's SIC division (from EDGAR).
        division = get_sic_division(get_sic_code(t))
        if division in SIC_DIVISIONS:
            X[i, division_order.index(division)] = 1.0

        stats = style_stats[date_str]

        if shares is not None and shares > 0:
            mcap = p_t * shares
            size_raw = np.log(max(mcap, 1e4))
            X[i, 10] = _z_score(size_raw, *stats["size"])
            if equity is not None:
                value_raw = equity / max(mcap, 1e4)
                X[i, 11] = _z_score(value_raw, *stats["value"])
            # If equity is unavailable, Value exposure stays 0.0 (neutral, not fabricated).

        mom_raw = rets_252.get(t, 0.0)
        if not pd.isna(mom_raw):
            X[i, 12] = _z_score(mom_raw, *stats["momentum"])
        vol_raw = vol_60.get(t, 0.15)
        if not pd.isna(vol_raw):
            X[i, 13] = _z_score(vol_raw, *stats["volatility"])

    return X
