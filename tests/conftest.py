"""
Shared test fixtures.

Keeps the test suite fast and deterministic/offline:
  1. Forces the memory system onto its deterministic pseudo-embedding fallback so tests never
     download the (self-hosted) BGE embedding model.
  2. Stubs the EDGAR 10-K filing text pipeline in the orchestrator so integration runs do not
     download filing documents (the graph wiring itself is exercised via dedicated tests that
     override these stubs with synthetic filings).
  3. Stubs the mechanical 300-name estimation universe builder in orchestrator integration runs so
     tests do not fetch the full 1500-name SEC registry from yfinance/EDGAR unless running the
     dedicated selection tests in test_estimation_universe.py.
  4. Provides offline fallbacks for external network calls (SEC EDGAR, Ken French RF data, yfinance).
"""
import pytest
import pandas as pd
import numpy as np
import src.orchestrator as orchestrator_mod
import src.memory.scoring as scoring
import src.data.edgar as edgar_mod
import src.data.barra_estimation as barra_est_mod
import src.data.prices as prices_mod


@pytest.fixture(scope="session", autouse=True)
def _use_fallback_embeddings():
    scoring._model_instance = "fallback"
    yield


@pytest.fixture(scope="session", autouse=True)
def _offline_environment_stubs():
    mp = pytest.MonkeyPatch()
    mp.setattr(orchestrator_mod, "get_10k_filing_documents", lambda ticker: [])
    mp.setattr(orchestrator_mod, "get_filing_text", lambda filing: "")
    mp.setattr(orchestrator_mod, "build_estimation_universe",
               lambda size=300: [f"EST{i:03d}" for i in range(size)])

    def default_orch_prices(tickers, start_date="2015-01-01", end_date="2023-12-31", **kw):
        dates = pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
        return pd.DataFrame(100.0, index=dates, columns=tickers)

    mp.setattr(orchestrator_mod, "get_pivot_close_prices", default_orch_prices)

    orig_pit = edgar_mod.get_point_in_time_fundamentals

    def safe_pit(ticker, decision_date=None, closing_price=None):
        try:
            return orig_pit(ticker, decision_date, closing_price)
        except Exception:
            return {
                "fundamentals_available": True,
                "shares_outstanding": 1e9,
                "stockholders_equity": 1e12,
                "sic": "7372",
                "reason": "Test fallback",
            }

    mp.setattr(edgar_mod, "get_point_in_time_fundamentals", safe_pit)

    # Safe RF series fallback
    rf_dates = pd.date_range("2015-01-01", "2025-01-01", freq="B").strftime("%Y-%m-%d")
    mp.setattr(barra_est_mod, "fetch_risk_free_rate", lambda *a, **kw: pd.Series(0.0001, index=rf_dates))

    yield
    mp.undo()
