"""
Mechanical construction of the locked 300-name Factor Estimation Universe (Step 1).

The estimation universe is the separate cross-section used for the daily Barra factor-return
regression (10 industry-division dummies + 4 style factors). It is NOT the traded 15-asset book:
it is selected mechanically from the complete SEC EDGAR registry (company_tickers.json, the
full ~10k-name list of SEC-registered filers) as follows

  1. Pool    = every registry ticker in SEC-published order (the registry is ordered by
               market capitalization, largest first), minus crypto/ETF book names and
               non-common-stock security classes ('-', '^').
  2. Filter  = a name must have a gap-free 2015-01-01..2023-12-31 price history and a
               point-in-time shares-outstanding value as of the selection reference date.
  3. Rank    = surviving names are ranked by point-in-time market capitalization on the
               reference date (reference close x point-in-time shares outstanding).
  4. Select  = the top `size` (300) names become the estimation universe.

The selection is cached to data/estimation_universe.json so the daily loop never re-runs the
mechanical selection; the cache records the method/version, size, and reference date so a
stale or mismatched cache is not silently reused. Nothing here is hardcoded: the same rule
over the same registry reproduces the same universe.
"""
import json

import pandas as pd

from config.settings import (
    DATA_DIR, FACTOR_ESTIMATION_UNIVERSE_SIZE, SELECTION_REF_DATE,
    PRICE_START_DATE, PRICE_END_DATE, CRYPTO_BOOK, ETF_BOOK,
)
from src.data import prices as prices_mod
from src.data.edgar import TICKERS_JSON_PATH, fetch_cik_mapping, get_point_in_time_fundamentals

ESTIMATION_UNIVERSE_CACHE_PATH = DATA_DIR / "estimation_universe.json"

METHOD_NAME = "sec_registry_pit_mcap_rank"

# Names required to qualify before the mechanical scan stops (margin over `size` so the
# top-`size` by market-cap rank is stable given the registry's market-cap ordering).
QUALIFYING_MARGIN_MULTIPLIER = 2.0
# Upper bound on how many registry names the scan may fetch prices for (protects a run from
# sweeping the entire ~10k registry when many early names fail the availability filters).
MAX_SCAN_CANDIDATES = 1500
PRICE_FETCH_CHUNK = 250


def _is_common_equity(ticker: str) -> bool:
    """Mechanical class filter: common-equity tickers only (no crypto/ETF/preferred/'^')."""
    t = ticker.upper()
    if t in CRYPTO_BOOK or t in ETF_BOOK:
        return False
    if "-" in t or "^" in t:
        return False
    return True


def _registry_tickers() -> list[str]:
    """Complete SEC registry ticker list, in the registry's published (market-cap) order."""
    cik_map = fetch_cik_mapping()
    with open(TICKERS_JSON_PATH, "r") as f:
        registry = json.load(f)
    pool = [str(entry["ticker"]).upper() for entry in registry.values() if _is_common_equity(str(entry["ticker"]))]
    seen = set()
    deduped = []
    for t in pool:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _batch_price_pivot(batch: list[str]) -> pd.DataFrame:
    """
    Reads the on-disk price cache for a batch and fetches (from yfinance) only the names not yet
    present. Unlike get_pivot_close_prices, this does not re-fetch a whole batch just because some
    of its names have partial history — partial-history names legitimately live in the cache but
    fail the full-range coverage guard.
    """
    df_cached = pd.read_parquet(prices_mod.PRICES_CACHE_PATH)
    present = set(df_cached["Ticker"].unique())
    missing = [t for t in batch if t not in present]
    if missing:
        fetched = prices_mod.fetch_and_cache_prices(missing, start_date=PRICE_START_DATE, end_date=PRICE_END_DATE)
        df_cached = pd.concat([df_cached, fetched]).drop_duplicates(subset=["Date", "Ticker"])
    pivot = (
        df_cached[df_cached["Ticker"].isin(batch)]
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
    )
    return pivot


def _names_with_full_history(
    pool: list[str], ref_date: str, size: int
) -> tuple[dict[str, float], int]:
    """
    Batch-fetches prices for the pool (in registry order) and keeps names with a gap-free
    full-range history plus a reference-date close. Returns (ticker -> ref_close) and the
    number of registry names scanned. Stops early once `size * QUALIFYING_MARGIN_MULTIPLIER`
    names qualify.
    """
    history = {}
    scanned = 0
    max_scan = min(len(pool), MAX_SCAN_CANDIDATES)
    for start in range(0, max_scan, PRICE_FETCH_CHUNK):
        batch = pool[start:start + PRICE_FETCH_CHUNK]
        scanned += len(batch)
        pivot = _batch_price_pivot(batch)
        for t in batch:
            if t not in pivot.columns:
                continue
            series = pivot[t].dropna()
            if series.empty:
                continue
            if series.index.min() > "2015-01-15" or series.index.max() < "2023-12-15":
                continue
            ref_sub = series[series.index >= ref_date]
            if ref_sub.empty:
                continue
            history[t] = float(ref_sub.iloc[0])
        if len(history) >= int(size * QUALIFYING_MARGIN_MULTIPLIER):
            break
    return history, scanned


def build_estimation_universe(
    size: int = FACTOR_ESTIMATION_UNIVERSE_SIZE,
    ref_date: str = SELECTION_REF_DATE,
    force_refresh: bool = False,
) -> list[str]:
    """
    Builds (or loads from cache) the mechanically-selected estimation universe.

    Returns a list of `size` tickers. Cached to data/estimation_universe.json; the cache is
    only reused when method, size, and reference date all match the current settings.
    """
    if ESTIMATION_UNIVERSE_CACHE_PATH.exists() and not force_refresh:
        cached = json.loads(ESTIMATION_UNIVERSE_CACHE_PATH.read_text())
        if (
            cached.get("method") == METHOD_NAME
            and cached.get("size") == size
            and cached.get("ref_date") == ref_date
        ):
            return cached["tickers"]

    pool = _registry_tickers()
    history, scanned = _names_with_full_history(pool, ref_date, size)

    scored = []
    for t, ref_close in history.items():
        fund = get_point_in_time_fundamentals(t, decision_date=ref_date, closing_price=ref_close)
        shares = fund.get("shares_outstanding")
        if not shares or shares <= 0:
            continue
        scored.append({"ticker": t, "market_cap": ref_close * shares})

    df_rank = pd.DataFrame(scored).sort_values("market_cap", ascending=False)
    selected = df_rank["ticker"].head(size).tolist()

    if len(selected) < size:
        raise RuntimeError(
            f"Only {len(selected)}/{size} estimation-universe names qualified after scanning "
            f"{scanned} registry candidates."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ESTIMATION_UNIVERSE_CACHE_PATH.write_text(json.dumps({
        "method": METHOD_NAME,
        "size": size,
        "ref_date": ref_date,
        "scanned": scanned,
        "tickers": selected,
    }, indent=2))

    return selected
