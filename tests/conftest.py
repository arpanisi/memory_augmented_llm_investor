"""
Shared test fixtures.

Keeps the test suite fast and deterministic/offline:
  1. Forces the memory system onto its deterministic pseudo-embedding fallback so tests never
     download the (self-hosted) BGE embedding model.
  2. Stubs the EDGAR 10-K filing text pipeline in the orchestrator so integration runs do not
     download filing documents (the graph wiring itself is exercised via dedicated tests that
     override these stubs with synthetic filings).
"""
import pytest
import src.orchestrator as orchestrator_mod
import src.memory.scoring as scoring
import src.data.edgar as edgar_mod


@pytest.fixture(scope="session", autouse=True)
def _use_fallback_embeddings():
    scoring._model_instance = "fallback"
    yield


@pytest.fixture(scope="session", autouse=True)
def _offline_edgar_filing_fetch():
    mp = pytest.MonkeyPatch()
    mp.setattr(orchestrator_mod, "get_10k_filing_documents", lambda ticker: [])
    mp.setattr(orchestrator_mod, "get_filing_text", lambda filing: "")
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
    yield
    mp.undo()

