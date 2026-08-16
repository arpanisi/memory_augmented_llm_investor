"""
Barra-style factor estimation (Step 1 in the locked plan).

Runs the offline estimation-universe daily cross-sectional regression: 10 industry-division
dummies + 4 style z-scores, with a market-cap-weighted industry-neutrality constraint (KKT
constrained WLS). The daily factor-return series is computed once and cached. The per-traded-asset
exposure lookup that consumes these daily stats lives in `barra_exposures`.

Fundamental inputs are point-in-time from SEC EDGAR; a missing value is never replaced with a
fabricated number — stocks without a valid (current or carried-forward) shares/equity value are
excluded from the day's cross-sectional regression, and a traded equity whose value is unavailable
gets a neutral (0.0) style exposure rather than an invented one.
"""
import io
import requests
import numpy as np
import pandas as pd
from scipy.stats import zscore
from config.settings import (
    DATA_DIR, PRICE_START_DATE, PRICE_END_DATE, SIC_DIVISIONS, EDGAR_USER_AGENT
)
from src.data.edgar import get_sic_code, build_pit_fundamentals_panel
from src.data.barra_exposures import get_sic_division, _compute_daily_stock_row

BARRA_CACHE_PATH = DATA_DIR / "barra_factor_returns.parquet"
RF_CACHE_PATH = DATA_DIR / "rf_daily.csv"

STYLE_COLUMNS = ["Size", "Value", "Momentum", "Volatility"]

# Raw cross-sectional columns (from _compute_daily_stock_row) backing each style factor.
STYLE_RAW_COLUMNS = {
    "Size": "size_raw",
    "Value": "value_raw",
    "Momentum": "mom_raw",
    "Volatility": "vol_raw",
}

# Momentum (252d) and Volatility (60d) need trailing history before the first regression day.
ESTIMATION_WARMUP_BARS = 252


def fetch_risk_free_rate(force_refresh: bool = False) -> pd.Series:
    """
    Downloads Ken French Fama/French 3-Factor daily file and returns RF daily series.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RF_CACHE_PATH.exists() and not force_refresh:
        df_rf = pd.read_csv(RF_CACHE_PATH)
        df_rf["Date"] = pd.to_datetime(df_rf["Date"]).dt.strftime("%Y-%m-%d")
        return df_rf.set_index("Date")["RF"]

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    print("Downloading Ken French RF data...")
    try:
        res = requests.get(url, headers={"User-Agent": EDGAR_USER_AGENT}, timeout=20)
        res.raise_for_status()
        import zipfile
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                lines = f.readlines()

        # Parse CSV lines
        data_lines = []
        parsing = False
        for line in lines:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("19") or decoded.startswith("20"):
                parts = [p.strip() for p in decoded.split(",")]
                if len(parts) >= 5 and len(parts[0]) == 8:
                    date_str = pd.to_datetime(parts[0], format="%Y%m%d").strftime("%Y-%m-%d")
                    rf_val = float(parts[4]) / 100.0  # Convert percentage to decimal
                    data_lines.append({"Date": date_str, "RF": rf_val})

        df_rf = pd.DataFrame(data_lines)
        df_rf.to_csv(RF_CACHE_PATH, index=False)
        return df_rf.set_index("Date")["RF"]
    except Exception as e:
        print(f"Warning: Failed to fetch Ken French RF data ({e}), falling back to 0.0001 daily RF default.")
        # Fallback daily RF of ~2.5% annualized if network blocked
        dates = pd.date_range(PRICE_START_DATE, PRICE_END_DATE, freq="B").strftime("%Y-%m-%d")
        df_rf = pd.DataFrame({"Date": dates, "RF": 0.0001})
        df_rf.to_csv(RF_CACHE_PATH, index=False)
        return df_rf.set_index("Date")["RF"]


def _collect_daily_valid_rows(
    estimation_tickers: list[str], rets_d, price_pivot: pd.DataFrame, decision_date,
    rets_252, vol_60, sic_map: dict, rf_val: float, pit_panel: dict | None = None
) -> list[dict]:
    """
    Assembles the day's valid estimation-universe cross-section. Stocks with missing/zero price or
    missing point-in-time shares/equity are excluded (never fabricated); a day with zero valid names
    is skipped by the caller because there is nothing to regress. `pit_panel` (from
    edgar.build_pit_fundamentals_panel) supplies per-ticker point-in-time fundamentals so the
    regression does not re-parse every companyfacts file for every name on every day.
    """
    valid_rows = []
    for t in estimation_tickers:
        r_it = rets_d.get(t)
        p_it = price_pivot.loc[decision_date, t] if t in price_pivot.columns else None
        if pd.isna(r_it) or pd.isna(p_it) or p_it <= 0:
            continue
        fund = pit_panel.get(t) if pit_panel else None
        row = _compute_daily_stock_row(
            t, r_it, p_it, rets_252, vol_60, sic_map, rf_val, decision_date, pit_fundamentals=fund
        )
        if row is not None:
            valid_rows.append(row)
    return valid_rows


def _style_stats_for_day(df_day: pd.DataFrame) -> dict:
    """
    Estimation-universe cross-sectional mean/std (population, ddof=0) for each style raw column.
    These are the normalization statistics used to z-score traded-equity style exposures.
    """
    return {
        "size": (float(df_day["size_raw"].mean()), float(df_day["size_raw"].std(ddof=0))),
        "value": (float(df_day["value_raw"].mean()), float(df_day["value_raw"].std(ddof=0))),
        "momentum": (float(df_day["mom_raw"].mean()), float(df_day["mom_raw"].std(ddof=0))),
        "volatility": (float(df_day["vol_raw"].mean()), float(df_day["vol_raw"].std(ddof=0))),
    }


def _add_style_z_scores(df_day: pd.DataFrame) -> None:
    """
    Computes cross-sectional style z-scores in place. A degenerate cross-section (std == 0, e.g. a
    single valid name) yields NaN from scipy.stats.zscore; those are treated as neutral (0.0)
    z-scores so the day's regression still runs instead of silently producing a gap.
    """
    for style in STYLE_COLUMNS:
        raw_col = STYLE_RAW_COLUMNS[style]
        df_day[style] = np.nan_to_num(zscore(df_day[raw_col]), nan=0.0)


def _add_industry_dummies(df_day: pd.DataFrame) -> list[str]:
    """
    Adds the 10 industry-division one-hot dummy columns in place and returns their column names.
    """
    for div in SIC_DIVISIONS.keys():
        df_day[f"Ind_{div}"] = (df_day["division"] == div).astype(float)
    return [f"Ind_{div}" for div in SIC_DIVISIONS.keys()]


def _constrained_wls_factor_returns(df_day: pd.DataFrame, reg_cols: list[str]) -> np.ndarray | None:
    """
    Runs the constrained WLS regression (market-cap-weighted) for the day and returns the factor
    returns, or None on failure (caller skips the day). Constraint C f = 0 enforces the
    market-cap-weighted sum of the 10 industry-division factors equals zero.
    """
    X = df_day[reg_cols].values
    y = df_day["excess_return"].values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    w = np.sqrt(df_day["mcap"].values)
    W = np.diag(w)

    try:
        # Market-cap weights applied to the industry-neutrality constraint.
        mcap_weights = df_day["mcap"].values / df_day["mcap"].sum()
        C = np.zeros((1, X.shape[1]))
        for j in range(10):
            C[0, j] = np.sum(mcap_weights * X[:, j])

        # Solve system via KKT matrix: [[X' W X, C'], [C, 0]] [f, lambda]' = [X' W y, 0]
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y

        KKT_top = np.hstack([XtWX, C.T])
        KKT_bot = np.hstack([C, np.zeros((1, 1))])
        KKT = np.vstack([KKT_top, KKT_bot])

        rhs = np.concatenate([XtWy, [0.0]])
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        return sol[:X.shape[1]]
    except Exception as e:
        return None


def compute_barra_model(
    price_pivot: pd.DataFrame, estimation_tickers: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Runs the daily cross-sectional Barra regression over estimation_tickers.

    Returns (factor_returns_df, style_stats):
      - factor_returns_df: DataFrame with a 'Date' column and 14 factor-return columns
        (10 Industry Division factors + 4 Style factors).
      - style_stats: dict mapping date -> {"size": (mean, std), "value": (mean, std),
        "momentum": (mean, std), "volatility": (mean, std)} where mean/std are the estimation
        universe's cross-sectional mean/std (population, ddof=0) for that day. These are the
        normalization statistics used to z-score traded-equity style exposures.

    No invented minimum-sample-size threshold: the regression runs with however many valid names
    exist that day (the constrained lstsq handles small cross-sections); a day with zero valid names
    is skipped because there is nothing to regress.
    """
    rf_series = fetch_risk_free_rate()
    dates = price_pivot.index
    daily_returns = price_pivot.pct_change()

    # Pre-map SIC division for estimation tickers from the EDGAR submissions index (companyfacts
    # does not carry SIC codes); the same source the traded-exposure path uses via get_sic_code.
    sic_map = {}
    for t in estimation_tickers:
        sic_map[t] = get_sic_division(get_sic_code(t))

    # Point-in-time fundamentals panel (shares/equity as of each regression day) built once for
    # the whole universe instead of per (name, day) — keeps the 300-name regression tractable.
    pit_panel = build_pit_fundamentals_panel(estimation_tickers, dates)

    factor_records = []
    style_stats = {}

    for idx, d in enumerate(dates):
        if idx < ESTIMATION_WARMUP_BARS:
            # Need trailing history for Momentum (252d) and Volatility (60d)
            continue

        rf_val = rf_series.get(d, 0.0001)
        rets_d = daily_returns.loc[d]

        # 252d momentum skipping recent 21d
        rets_252 = (price_pivot.iloc[idx - 21] / price_pivot.iloc[idx - 252]) - 1.0
        # 60d volatility
        vol_60 = daily_returns.iloc[idx - 60:idx].std() * np.sqrt(252)

        valid_rows = _collect_daily_valid_rows(
            estimation_tickers, rets_d, price_pivot, d, rets_252, vol_60, sic_map, rf_val, pit_panel
        )
        if not valid_rows:
            # Nothing to regress on this day (no data, not an invented threshold).
            continue

        df_day = pd.DataFrame(valid_rows)
        # Cross-sectional mean/std (population, ddof=0) for traded-equity z-scoring later.
        style_stats[str(d)] = _style_stats_for_day(df_day)

        # Build the design matrix: style z-scores then industry dummies.
        _add_style_z_scores(df_day)
        ind_cols = _add_industry_dummies(df_day)
        reg_cols = ind_cols + STYLE_COLUMNS

        f_returns = _constrained_wls_factor_returns(df_day, reg_cols)
        if f_returns is None:
            continue

        rec = {"Date": d}
        for idx_col, col_name in enumerate(reg_cols):
            rec[col_name] = f_returns[idx_col]
        factor_records.append(rec)

    df_factors = pd.DataFrame(factor_records)
    if not df_factors.empty:
        df_factors.to_parquet(BARRA_CACHE_PATH, index=False)
    return df_factors, style_stats


def compute_barra_daily_factors(price_pivot: pd.DataFrame, estimation_tickers: list[str]) -> pd.DataFrame:
    """
    Public wrapper returning only the daily factor-return time series.
    """
    df_factors, _ = compute_barra_model(price_pivot, estimation_tickers)
    return df_factors
