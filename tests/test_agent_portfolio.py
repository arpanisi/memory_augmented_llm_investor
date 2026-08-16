"""
Unit tests for Steps 3, 4, and 5: View Formation, Feedback Loop, and Portfolio Optimization.
"""
import pytest
import numpy as np
import pandas as pd
from src.agent.prompts import format_view_formation_user_prompt
from src.agent.view_formation import parse_view_formation_response, call_view_formation_agent
from src.memory.store import AssetMemoryStore
from src.memory.feedback import FeedbackTracker
from src.portfolio.shrinkage import compute_ledoit_wolf_constant_correlation
from src.portfolio.optimizer import solve_portfolio_optimization

def test_view_formation_response_parsing():
    raw_json = '{"direction": "long", "conviction": 0.85, "rationale": "Strong momentum."}'
    parsed = parse_view_formation_response(raw_json)
    assert parsed["direction"] == "long"
    assert parsed["conviction"] == pytest.approx(0.85)
    assert "momentum" in parsed["rationale"]

def test_prompt_formatting_with_sentinel():
    fund_sentinel = {"fundamentals_available": False, "reason": "Crypto asset"}
    prompt = format_view_formation_user_prompt(
        "BTC-USD", "2020-01-01", [{"Date": "2019-12-31", "Close": 7200.0}], fund_sentinel, []
    )
    assert "fundamentals_available: false" in prompt
    assert "Crypto asset" in prompt

def test_prompt_includes_relationships_summary():
    """Finding 1: the company-relationship summary is threaded into the view-formation prompt, and an
    explicit empty-set indicator is used when there are no edges (never an omitted field)."""
    fund = {"fundamentals_available": True, "pe_ratio": 20.0, "pb_ratio": 5.0,
            "net_income": 1e9, "stockholders_equity": 1e10}
    prompt = format_view_formation_user_prompt(
        "AAPL", "2020-01-01", [{"Date": "2019-12-31", "Close": 100.0}], fund, [],
        relationships_summary="named competitors: MSFT, GOOGL; named suppliers: TSLA"
    )
    assert "Company Relationships" in prompt
    assert "MSFT" in prompt and "GOOGL" in prompt and "TSLA" in prompt

    prompt_empty = format_view_formation_user_prompt(
        "AAPL", "2020-01-01", [{"Date": "2019-12-31", "Close": 100.0}], fund, [],
        relationships_summary="No company relationships on record."
    )
    assert "No company relationships on record" in prompt_empty

def test_feedback_loop_tracker():
    tracker = FeedbackTracker(lookback_days=5)
    store = AssetMemoryStore("AAPL")
    mem = store.add_memory("Earnings beat", "2020-01-01", layer="short", importance=50.0)

    trading_days = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    for d in trading_days:
        tracker.record_decision(d, "AAPL", [mem.memory_id], "long", 100.0)

    # Price pivot with +10% price return over the window
    dates = pd.to_datetime(trading_days)
    prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    df_price = pd.DataFrame({"AAPL": prices}, index=trading_days)

    tracker.process_feedback_for_day("2020-01-08", {"AAPL": store}, df_price)
    
    # Positive return -> importance should increase by +18 (50 -> 68)
    assert mem.importance == pytest.approx(68.0)

def test_feedback_applies_in_orchestrator_call_order():
    """Finding 3: feedback must fire when process_feedback_for_day(d) runs BEFORE day d's own
    decision is recorded (the real orchestrator call order). Previously the guard clause returned
    early every time and no memory was ever updated."""
    tracker = FeedbackTracker(lookback_days=5)
    store = AssetMemoryStore("AAPL")
    mem = store.add_memory("Earnings beat", "2020-01-01", layer="short", importance=50.0)

    trading_days = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]
    prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    df_price = pd.DataFrame({"AAPL": prices}, index=trading_days)

    # Orchestrator call order: process feedback for today, then record today's decision afterwards.
    for i, d in enumerate(trading_days):
        tracker.process_feedback_for_day(d, {"AAPL": store}, df_price)
        tracker.record_decision(d, "AAPL", [mem.memory_id], "long", prices[i])

    # The decision made 5 trading days earlier (day 0) earned +10% -> +18 importance.
    assert mem.importance == pytest.approx(68.0)

def test_ledoit_wolf_shrinkage_properties():
    np.random.seed(42)
    T, N = 252, 5
    returns = np.random.randn(T, N) * 0.02
    Sigma_shrunk, delta = compute_ledoit_wolf_constant_correlation(returns)
    
    assert Sigma_shrunk.shape == (N, N)
    assert 0.0 <= delta <= 1.0
    # Check symmetric
    assert np.allclose(Sigma_shrunk, Sigma_shrunk.T)
    # Check positive definite
    eigvals = np.linalg.eigvalsh(Sigma_shrunk)
    assert np.all(eigvals > 0)

def test_portfolio_optimizer_constraints():
    np.random.seed(42)
    N = 4
    asset_names = ["AAPL", "MSFT", "BTC-USD", "SPY"]
    mean_returns = np.array([0.001, 0.0008, 0.002, 0.0005])
    
    returns = np.random.randn(252, N) * 0.015
    Sigma_shrunk, _ = compute_ledoit_wolf_constant_correlation(returns)

    views = [
        {"direction": "long", "conviction": 0.4},
        {"direction": "short", "conviction": 0.3},
        {"direction": "flat", "conviction": 0.0},
        {"direction": "long", "conviction": 0.5},
    ]

    # Factor exposures: N x 14
    X = np.random.randn(N, 14) * 0.1

    w = solve_portfolio_optimization(
        asset_names, mean_returns, Sigma_shrunk, views, X, risk_aversion_lambda=1.0
    )

    assert len(w) == N
    # Budget constraint ||w||_1 <= 1
    assert np.sum(np.abs(w)) <= 1.0 + 1e-5

    # Direction and conviction bounds
    assert 0.0 - 1e-5 <= w[0] <= 0.4 + 1e-5    # AAPL long <= 0.4
    assert -0.3 - 1e-5 <= w[1] <= 0.0 + 1e-5   # MSFT short >= -0.3
    assert abs(w[2]) <= 1e-5            # BTC-USD flat == 0
    assert 0.0 - 1e-5 <= w[3] <= 0.5 + 1e-5    # SPY long <= 0.5

    # Barra factor exposure bounds
    for k in range(14):
        exposure_k = np.dot(X[:, k], w)
        bound_k = 0.5 if k < 10 else 0.3
        assert abs(exposure_k) <= bound_k + 1e-4

def test_all_flat_day_fallback():
    asset_names = ["AAPL", "MSFT"]
    views = [{"direction": "flat"}, {"direction": "flat"}]
    w = solve_portfolio_optimization(
        asset_names, np.array([0.01, 0.01]), np.eye(2), views, np.zeros((2, 14))
    )
    assert np.all(w == 0.0)
