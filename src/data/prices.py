"""
Price data pipeline using yfinance with local caching and NYSE calendar alignment.
"""
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from config.settings import DATA_DIR, PRICE_START_DATE, PRICE_END_DATE

PRICES_CACHE_PATH = DATA_DIR / "prices_cache.parquet"

# A freshly-fetched cache can legitimately start a few calendar days after `start_date` (when the
# start date is a weekend/holiday) and ends strictly before `end_date` (yfinance's end is exclusive).
# Allow a small tolerance so genuinely full caches are reused, while a cache missing weeks/months/
# years of the requested range (the stale-cache defect) still triggers a re-fetch.
CACHE_COVERAGE_TOLERANCE_DAYS = 7

def _cache_covers_date_range(df_cached: pd.DataFrame, tickers: list[str], start_date: str, end_date: str) -> bool:
    """
    True only if the cached rows for EVERY requested ticker actually span the full requested
    date range [start_date, end_date]. A ticker merely being present somewhere in the cache is
    not enough: a cache written by an earlier narrow-range fetch must not be silently reused for
    a wider request (that previously left BTC-USD/SPY almost entirely missing).
    """
    cached_tickers = set(df_cached["Ticker"].unique())
    if not all(t in cached_tickers for t in tickers):
        return False
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    min_allowed = start + pd.Timedelta(days=CACHE_COVERAGE_TOLERANCE_DAYS)
    max_allowed = end - pd.Timedelta(days=CACHE_COVERAGE_TOLERANCE_DAYS)
    for t in tickers:
        sub = df_cached[df_cached["Ticker"] == t]["Date"]
        if sub.empty:
            return False
        if pd.Timestamp(sub.min()) > min_allowed:
            return False
        if pd.Timestamp(sub.max()) < max_allowed:
            return False
    return True

def fetch_and_cache_prices(tickers: list[str], start_date: str = PRICE_START_DATE, end_date: str = PRICE_END_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch daily OHLCV prices for a list of tickers from yfinance and cache locally.
    Returns a MultiIndex DataFrame (Date, Ticker) or wide DataFrame of Close prices.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if PRICES_CACHE_PATH.exists() and not force_refresh:
        df_cached = pd.read_parquet(PRICES_CACHE_PATH)
        if _cache_covers_date_range(df_cached, tickers, start_date, end_date):
            return df_cached[df_cached["Ticker"].isin(tickers)]

    print(f"Fetching price data for {len(tickers)} tickers from yfinance...")
    raw_data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", auto_adjust=True, progress=False)
    
    records = []
    if len(tickers) == 1:
        ticker = tickers[0]
        df_t = raw_data.dropna(how="all")
        for idx, row in df_t.iterrows():
            records.append({
                "Date": pd.to_datetime(idx).strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Open": float(row.get("Open", row.get("Close", 0.0))),
                "High": float(row.get("High", row.get("Close", 0.0))),
                "Low": float(row.get("Low", row.get("Close", 0.0))),
                "Close": float(row.get("Close", 0.0)),
                "Volume": float(row.get("Volume", 0.0)),
            })
    else:
        for ticker in tickers:
            if ticker in raw_data.columns.levels[0]:
                df_t = raw_data[ticker].dropna(how="all")
                for idx, row in df_t.iterrows():
                    records.append({
                        "Date": pd.to_datetime(idx).strftime("%Y-%m-%d"),
                        "Ticker": ticker,
                        "Open": float(row.get("Open", row.get("Close", 0.0))),
                        "High": float(row.get("High", row.get("Close", 0.0))),
                        "Low": float(row.get("Low", row.get("Close", 0.0))),
                        "Close": float(row.get("Close", 0.0)),
                        "Volume": float(row.get("Volume", 0.0)),
                    })

    df = pd.DataFrame(records)
    if not df.empty:
        if PRICES_CACHE_PATH.exists() and not force_refresh:
            existing_df = pd.read_parquet(PRICES_CACHE_PATH)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=["Date", "Ticker"])
            combined_df.to_parquet(PRICES_CACHE_PATH, index=False)
            return combined_df[combined_df["Ticker"].isin(tickers)]
        else:
            df.to_parquet(PRICES_CACHE_PATH, index=False)
            return df

    return df

def get_pivot_close_prices(tickers: list[str], start_date: str = PRICE_START_DATE, end_date: str = PRICE_END_DATE) -> pd.DataFrame:
    """
    Returns a wide DataFrame of Close prices indexed by Date, columns = Tickers.
    """
    df = fetch_and_cache_prices(tickers, start_date=start_date, end_date=end_date)
    pivot = df.pivot(index="Date", columns="Ticker", values="Close")
    return pivot.sort_index()

def check_history_continuity(ticker: str, start_date: str = PRICE_START_DATE, end_date: str = PRICE_END_DATE) -> bool:
    """
    Checks if a ticker has unbroken price history across the span.
    """
    pivot = get_pivot_close_prices([ticker], start_date, end_date)
    if ticker not in pivot.columns:
        return False
    series = pivot[ticker].dropna()
    if series.empty:
        return False
    # Check start and end dates coverage
    min_date = series.index.min()
    max_date = series.index.max()
    if min_date > "2015-01-15" or max_date < "2023-12-15":
        return False
    return True
