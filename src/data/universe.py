"""
Universe selection module enforcing the mechanical top-10 equities selection rule
plus fixed crypto and ETF books, creating the unified 15-asset universe.
"""
import pandas as pd
from config.settings import (
    CANDIDATE_POOL, EQUITIES_BOOK_SIZE, CRYPTO_BOOK, ETF_BOOK,
    SELECTION_REF_DATE, PRICE_START_DATE, PRICE_END_DATE
)
from src.data.prices import fetch_and_cache_prices, check_history_continuity, get_pivot_close_prices
from src.data.edgar import get_point_in_time_fundamentals

def get_reference_trading_date(ref_date: str = SELECTION_REF_DATE) -> str:
    """
    Returns the first trading day on or after ref_date.
    """
    df = get_pivot_close_prices(["SPY"], start_date=ref_date, end_date="2018-01-15")
    valid_dates = df.dropna(subset=["SPY"]).index
    return str(valid_dates[0])

def select_equities_book(candidate_pool: list[str] = CANDIDATE_POOL, size: int = EQUITIES_BOOK_SIZE) -> list[str]:
    """
    Ranks candidates by market cap on the reference trading day (price x EDGAR point-in-time shares)
    filtered by gap-free yfinance history from 2015-01-01 to 2023-12-31.
    Returns the top N equity tickers.
    """
    ref_date = get_reference_trading_date(SELECTION_REF_DATE)
    close_prices = get_pivot_close_prices(candidate_pool, start_date=PRICE_START_DATE, end_date=PRICE_END_DATE)
    
    candidate_scores = []
    for ticker in candidate_pool:
        # Check gap-free history 2015-2023
        if ticker not in close_prices.columns:
            continue
        series = close_prices[ticker].dropna()
        if series.empty or series.index.min() > "2015-01-15" or series.index.max() < "2023-12-15":
            continue
            
        p_ref = series.loc[series.index >= ref_date].iloc[0] if not series.loc[series.index >= ref_date].empty else None
        if p_ref is None or p_ref <= 0:
            continue

        fund = get_point_in_time_fundamentals(ticker, decision_date=ref_date, closing_price=p_ref)
        shares = fund.get("shares_outstanding")

        # get_point_in_time_fundamentals already resolves to the nearest prior filing's shares
        # count (it takes the latest CommonStockSharesOutstanding/EntityCommonStockSharesOutstanding
        # value among filings filed on or before decision_date). If shares is still missing here, no
        # filing ever reported it as of the reference date, so a true market cap cannot be computed
        # for this candidate -- exclude it from market-cap ranking rather than substitute an
        # arbitrary constant (which would silently rank the candidate by price alone, since
        # mcap = price * constant).
        if not shares or shares <= 0:
            continue

        mcap = p_ref * shares
        candidate_scores.append({"ticker": ticker, "market_cap": mcap})

    df_rank = pd.DataFrame(candidate_scores).sort_values("market_cap", ascending=False)
    selected = df_rank["ticker"].head(size).tolist()
    return selected

def build_full_universe() -> dict:
    """
    Returns the unified 15-asset portfolio specification containing equities, crypto, and ETF books.
    """
    equities = select_equities_book()
    full_universe = equities + CRYPTO_BOOK + ETF_BOOK
    return {
        "equities": equities,
        "crypto": CRYPTO_BOOK,
        "etfs": ETF_BOOK,
        "full_universe": full_universe,
        "total_assets": len(full_universe)
    }
