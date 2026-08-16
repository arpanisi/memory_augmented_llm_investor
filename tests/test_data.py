"""
Unit tests for Step 1 Data Pipeline components.
"""
import types

import pytest
import pandas as pd
import numpy as np
from config.settings import CANDIDATE_POOL, CRYPTO_BOOK, ETF_BOOK
from src.data.universe import build_full_universe, select_equities_book
from src.data.edgar import get_point_in_time_fundamentals
from src.data.prices import get_pivot_close_prices
from src.data.barra_exposures import get_sic_division

def test_universe_selection_size():
    """Verify universe has exactly 15 assets (10 equities + 2 crypto + 3 ETFs)."""
    univ = build_full_universe()
    assert univ["total_assets"] == 15
    assert len(univ["equities"]) == 10
    assert len(univ["crypto"]) == 2
    assert len(univ["etfs"]) == 3
    assert set(univ["crypto"]) == set(CRYPTO_BOOK)
    assert set(univ["etfs"]) == set(ETF_BOOK)

def test_edgar_sentinel_for_crypto_and_etfs():
    """Verify Crypto and ETF tickers return the explicit fundamentals_available: False sentinel."""
    for crypto_ticker in CRYPTO_BOOK:
        res = get_point_in_time_fundamentals(crypto_ticker, decision_date="2020-01-01")
        assert res["fundamentals_available"] is False
        assert "EDGAR" in res["reason"] or "filings" in res["reason"]

    for etf_ticker in ETF_BOOK:
        res = get_point_in_time_fundamentals(etf_ticker, decision_date="2020-01-01")
        assert res["fundamentals_available"] is False

def test_sic_division_mapping():
    """Verify static SIC code mapping to 10 Divisions A-J."""
    assert get_sic_division("0111") == "A"
    assert get_sic_division("1000") == "B"
    assert get_sic_division("1520") == "C"
    assert get_sic_division("2834") == "D"
    assert get_sic_division("4911") == "E"
    assert get_sic_division("5012") == "F"
    assert get_sic_division("5411") == "G"
    assert get_sic_division("6021") == "H"
    assert get_sic_division("7372") == "I"
    assert get_sic_division("9100") == "J"

def test_price_pivot_structure(monkeypatch, tmp_path):
    """Verify pivot table format and date alignment."""
    import src.data.prices as prices_mod
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-01", "2020-01-31", freq="B")]
    _write_price_cache(monkeypatch, tmp_path, ["AAPL", "BTC-USD", "SPY"], dates)
    tickers = ["AAPL", "BTC-USD", "SPY"]
    pivot = get_pivot_close_prices(tickers, start_date="2020-01-01", end_date="2020-01-31")
    assert not pivot.empty
    for t in tickers:
        assert t in pivot.columns


def _write_price_cache(monkeypatch, tmp_path, tickers, dates):
    """Writes a long-format price cache (Date, Ticker, ...) for the given tickers/dates.

    Uses monkeypatch so the module-level PRICES_CACHE_PATH is restored after the test,
    preventing the temporary cache from leaking into other tests in the same session.
    """
    import src.data.prices as prices_mod
    rows = []
    for t in tickers:
        for d in dates:
            rows.append({"Date": d, "Ticker": t, "Open": 100.0, "High": 101.0,
                         "Low": 99.0, "Close": 100.0, "Volume": 1000.0})
    cache_path = tmp_path / "prices_cache.parquet"
    pd.DataFrame(rows).to_parquet(cache_path, index=False)
    monkeypatch.setattr(prices_mod, "PRICES_CACHE_PATH", cache_path)
    return cache_path


def _yfinance_download_frame(tickers, dates):
    """Mimics yfinance's multi-ticker multi-column download DataFrame format."""
    tickers = list(tickers)
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    frame = pd.DataFrame(np.nan, index=pd.DatetimeIndex(dates), columns=cols)
    for t in tickers:
        frame.loc[:, (t, "Open")] = 100.0
        frame.loc[:, (t, "High")] = 101.0
        frame.loc[:, (t, "Low")] = 99.0
        frame.loc[:, (t, "Close")] = 100.0
        frame.loc[:, (t, "Volume")] = 1000.0
    return frame


def test_cache_coverage_check_refetches_when_cached_range_too_narrow(monkeypatch, tmp_path):
    """Regression for the stale-price-cache defect (code-review Finding 1): a cache whose rows for
    some tickers only cover a narrow date range must trigger a re-fetch when a wider range is
    requested, not silently return the partial cache."""
    import src.data.prices as prices_mod

    narrow_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-01", periods=5, freq="B")]
    _write_price_cache(monkeypatch, tmp_path, ["AAPL", "BTC-USD", "SPY"], narrow_dates)

    full_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2015-01-01", periods=5, freq="B")]
    calls = []
    fake_yf = types.SimpleNamespace(
        download=lambda tickers, start, end, **kwargs: (
            calls.append((list(tickers), start, end)) or
            _yfinance_download_frame(tickers, full_dates)
        )
    )
    monkeypatch.setattr(prices_mod, "yf", fake_yf)

    df = prices_mod.fetch_and_cache_prices(["AAPL", "BTC-USD", "SPY"], "2015-01-01", "2018-01-16")
    assert len(calls) == 1, "narrow-range cache must trigger exactly one re-fetch"

    pivot = df.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    for t in ["AAPL", "BTC-USD", "SPY"]:
        series = pivot[t].dropna()
        assert series.index.min() == full_dates[0], f"{t} must be re-fetched back to 2015"
        assert set(full_dates) <= set(series.index), f"{t} must actually contain the wider range"


def test_cache_reused_when_coverage_is_sufficient(monkeypatch, tmp_path):
    """A cache whose date range fully covers the requested range is reused without a re-fetch."""
    import src.data.prices as prices_mod

    wide_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2015-01-01", periods=10, freq="B")]
    _write_price_cache(monkeypatch, tmp_path, ["AAPL", "BTC-USD", "SPY"], wide_dates)

    calls = []
    fake_yf = types.SimpleNamespace(
        download=lambda tickers, start, end, **kwargs: calls.append((list(tickers), start, end))
    )
    monkeypatch.setattr(prices_mod, "yf", fake_yf)

    df = prices_mod.fetch_and_cache_prices(["AAPL", "BTC-USD", "SPY"], "2015-01-02", "2015-01-14")
    assert len(calls) == 0, "sufficiently-covered cache must not re-fetch"
    assert len(df) == 3 * 10
