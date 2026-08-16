"""
Unit tests for Steps 6, 7, and 8: Reporting, Evaluation Metrics, Baselines, and Orchestration Loop.
"""
import pytest
import numpy as np
import pandas as pd
from src.reporting.metrics import compute_portfolio_breadth, compute_realized_factor_exposures
from src.evaluation.metrics import (
    calculate_annualized_sharpe, calculate_cumulative_return, calculate_max_drawdown, compute_evaluation_summary
)
from src.evaluation.baselines import run_buy_and_hold_baseline, run_equal_weight_baseline
from src.orchestrator import DailyOrchestrator, derive_decision_calendar

def test_portfolio_breadth_uncorrelated_sanity_check():
    """When R is Identity matrix and w is equal-weighted across N assets, Breadth MUST equal N."""
    N = 5
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    Sigma_identity = np.eye(N)
    breadth = compute_portfolio_breadth(w, Sigma_identity)
    assert breadth == pytest.approx(5.0)

def test_portfolio_breadth_all_cash():
    """All cash portfolio (w = 0) returns None for Breadth."""
    breadth = compute_portfolio_breadth(np.zeros(5), np.eye(5))
    assert breadth is None

def test_realized_factor_exposures():
    w = np.array([0.5, -0.5])
    X = np.zeros((2, 14))
    X[0, 0] = 1.0   # Ind_A for asset 0
    X[1, 0] = 0.0   # Ind_A for asset 1
    X[0, 10] = 1.5  # Style_Size for asset 0
    X[1, 10] = -0.5 # Style_Size for asset 1

    realized = compute_realized_factor_exposures(w, X)
    assert realized["Ind_A"] == pytest.approx(0.5)
    # Style_Size = 0.5 * 1.5 + (-0.5) * (-0.5) = 0.75 + 0.25 = 1.0
    assert realized["Style_Size"] == pytest.approx(1.0)

def test_evaluation_metrics():
    # 10 days of +1% daily return
    rets = pd.Series([0.01] * 10)
    sr = calculate_annualized_sharpe(rets)
    cum_ret = calculate_cumulative_return(rets)
    mdd = calculate_max_drawdown(rets)

    assert cum_ret == pytest.approx((1.01**10) - 1.0)
    assert mdd == pytest.approx(0.0)  # Continuous positive return -> no drawdown

def test_baselines_attribution():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    df_prices = pd.DataFrame({
        "AAPL": [100.0, 102.0, 104.0, 106.0, 108.0],
        "MSFT": [200.0, 204.0, 208.0, 212.0, 216.0]
    }, index=dates)

    bh_rets = run_buy_and_hold_baseline(df_prices)
    ew_rets = run_equal_weight_baseline(df_prices)

    assert len(bh_rets) == 4
    assert len(ew_rets) == 4
    # Equal weight 2% daily return for both assets -> 2% daily portfolio return
    assert ew_rets.iloc[0] == pytest.approx(0.02)

def test_tier1_orchestration_run(monkeypatch):
    """Smoke test running Tier 1 daily loop across 3 assets and 3 days."""
    import src.orchestrator as orch_mod
    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL", "BTC-USD", "SPY"]
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2015-01-01", periods=800)]
    price_pivot = pd.DataFrame(100.0, index=dates, columns=["AAPL", "BTC-USD", "SPY"])
    monkeypatch.setattr(orch_mod, "get_pivot_close_prices", lambda *args, **kwargs: price_pivot)
    monkeypatch.setattr(orch_mod, "compute_barra_model", lambda p, t: (pd.DataFrame(), {}))
    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures", lambda *a, **k: np.zeros((3, 14)))
    monkeypatch.setattr(orch_mod, "call_view_formation_agent",
                        lambda **kw: {"direction": "flat", "conviction": 0.0, "rationale": "test"})

    # Run daily loop over short date range
    df_res = orch.run_daily_loop(start_date="2018-01-02", end_date="2018-01-05", is_warmup=True)
    assert not df_res.empty
    assert "Portfolio_Return" in df_res.columns
    assert "Breadth" in df_res.columns

def test_decision_calendar_matches_known_nyse_trading_days():
    """Decision days come from an equity/ETF calendar, not the crypto-inclusive date union."""
    calendar_days = pd.date_range("2018-01-02", "2018-01-16", freq="D").strftime("%Y-%m-%d").tolist()
    expected_nyse_days = [
        "2018-01-02",
        "2018-01-03",
        "2018-01-04",
        "2018-01-05",
        "2018-01-08",
        "2018-01-09",
        "2018-01-10",
        "2018-01-11",
        "2018-01-12",
        "2018-01-16",
    ]
    price_pivot = pd.DataFrame(index=calendar_days, columns=["AAPL", "BTC-USD", "SPY"], dtype=float)
    price_pivot["BTC-USD"] = np.arange(len(calendar_days), dtype=float) + 1000.0
    price_pivot.loc[expected_nyse_days, "AAPL"] = np.arange(len(expected_nyse_days), dtype=float) + 100.0
    price_pivot.loc[expected_nyse_days, "SPY"] = np.arange(len(expected_nyse_days), dtype=float) + 250.0

    decision_days = derive_decision_calendar(
        price_pivot=price_pivot,
        assets=["AAPL", "BTC-USD", "SPY"],
        reference_assets=["SPY"],
    )

    assert decision_days == expected_nyse_days
    assert "2018-01-06" not in decision_days
    assert "2018-01-07" not in decision_days
    assert "2018-01-13" not in decision_days
    assert "2018-01-14" not in decision_days
    assert "2018-01-15" not in decision_days

def test_convert_weights_to_shares():
    """Finding 6: target weights are converted into actual share counts using day-t closing prices."""
    from src.orchestrator import convert_weights_to_shares
    shares = convert_weights_to_shares(np.array([0.5, 0.25, 0.0]), 100.0, np.array([100.0, 50.0, 0.0]))
    assert shares[0] == pytest.approx(0.5)   # 0.5 * 100 / 100
    assert shares[1] == pytest.approx(0.5)   # 0.25 * 100 / 50
    assert shares[2] == pytest.approx(0.0)   # non-positive close price -> no shares

def test_share_conversion_tracks_portfolio_value():
    """Finding 6: converting to shares lets the portfolio value evolve day over day."""
    from src.orchestrator import convert_weights_to_shares
    shares = convert_weights_to_shares(np.array([0.5, 0.5]), 1000.0, np.array([100.0, 200.0]))
    assert shares[0] == pytest.approx(5.0)   # 0.5 * 1000 / 100
    assert shares[1] == pytest.approx(2.5)   # 0.5 * 1000 / 200
    value_next = float(np.sum(shares * np.array([110.0, 190.0])))
    assert value_next == pytest.approx(1025.0)  # +2.5% on 1000

def test_all_cash_day_returns_zero(monkeypatch):
    """Finding 6: an all-cash day (zero weights) must report a 0% portfolio return, not -100% — the
    uninvested budget is held as cash and carried forward."""
    import src.orchestrator as orch_mod
    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL", "BTC-USD", "SPY"]

    monkeypatch.setattr(orch_mod, "compute_barra_model", lambda p, t: (pd.DataFrame(), {}))
    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures",
                        lambda *a, **k: np.zeros((3, 14)))

    def fake_view(**kwargs):
        return {"direction": "flat", "conviction": 0.0, "rationale": "all flat"}
    monkeypatch.setattr(orch_mod, "call_view_formation_agent", fake_view)

    df_res = orch.run_daily_loop(start_date="2018-01-02", end_date="2018-01-03", is_warmup=True)
    assert not df_res.empty
    assert df_res["Portfolio_Return"].notna().all()
    assert all(abs(v) < 1e-9 for v in df_res["Portfolio_Return"].tolist())
    assert df_res["Gross_Exposure"].max() < 1e-6

def test_orchestrator_passes_real_exposures_to_optimizer(monkeypatch):
    """Finding 2: the hardcoded all-zero exposure matrix is replaced — the real (computed) exposures
    flow into both the optimizer and the realized factor-exposure report."""
    import src.orchestrator as orch_mod
    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL", "BTC-USD", "SPY"]

    monkeypatch.setattr(orch_mod, "compute_barra_model", lambda p, t: (pd.DataFrame(), {}))
    fake_X = np.zeros((3, 14))
    fake_X[0, 0] = 0.9    # AAPL Ind_A
    fake_X[0, 10] = 1.5   # AAPL Style_Size
    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures", lambda *a, **k: fake_X)

    captured = {}
    def fake_opt(**kwargs):
        captured["factor_exposures"] = kwargs["factor_exposures"]
        return np.array([0.1, 0.0, 0.2])
    monkeypatch.setattr(orch_mod, "solve_portfolio_optimization", fake_opt)

    df_res = orch.run_daily_loop(start_date="2018-01-02", end_date="2018-01-02", is_warmup=True)
    assert not df_res.empty
    assert captured.get("factor_exposures") is not None
    assert np.array_equal(captured["factor_exposures"], fake_X)
    # Realized factor exposure report uses the same real exposures: Ind_A = 0.1*0.9.
    assert df_res.iloc[0]["Factor_Exposures"]["Ind_A"] == pytest.approx(0.09)


def test_orchestrator_uses_prior_close_for_day_t_factor_exposures(monkeypatch):
    """Day-t optimizer inputs must use factor exposures available at t-1, not day-t close data."""
    import src.orchestrator as orch_mod

    assets = ["AAPL", "MSFT"]
    dates = pd.bdate_range("2016-12-30", periods=260).strftime("%Y-%m-%d").tolist()
    prev_d = dates[252]
    decision_d = dates[253]

    monkeypatch.setattr(orch_mod, "compute_barra_model", lambda p, t: (pd.DataFrame(), {prev_d: {}}))
    monkeypatch.setattr(
        orch_mod,
        "call_view_formation_agent",
        lambda **kwargs: {"direction": "flat", "conviction": 0.0, "rationale": "test"},
    )

    captured = []

    def fake_compute_asset_factor_exposures(price_pivot, estimation_universe, traded_equities, decision_date, style_stats):
        X = np.zeros((len(traded_equities), 14))
        X[0, 10] = float(price_pivot.loc[decision_date, "AAPL"])
        captured.append((decision_date, X.copy()))
        return X

    def fake_solve_portfolio_optimization(**kwargs):
        captured.append(("optimizer", kwargs["factor_exposures"].copy()))
        return np.zeros(len(kwargs["asset_names"]))

    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures", fake_compute_asset_factor_exposures)
    monkeypatch.setattr(orch_mod, "solve_portfolio_optimization", fake_solve_portfolio_optimization)

    def run_with_day_t_close(day_t_close):
        price_pivot = pd.DataFrame(100.0, index=dates, columns=assets)
        price_pivot.loc[decision_d, "AAPL"] = day_t_close
        monkeypatch.setattr(orch_mod, "get_pivot_close_prices", lambda *args, **kwargs: price_pivot)

        orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
        orch.assets = assets
        orch.run_daily_loop(start_date=decision_d, end_date=decision_d, is_warmup=True)

    run_with_day_t_close(100.0)
    run_with_day_t_close(1000.0)

    exposure_calls = [entry for entry in captured if entry[0] != "optimizer"]
    optimizer_calls = [entry for entry in captured if entry[0] == "optimizer"]

    assert [entry[0] for entry in exposure_calls] == [prev_d, prev_d]
    assert optimizer_calls[0][1][0, 10] == pytest.approx(100.0)
    assert optimizer_calls[1][1][0, 10] == pytest.approx(100.0)

def test_orchestrator_wires_relationship_graph_into_view_formation(monkeypatch):
    """Finding 1: newly-filed 10-Ks drive extraction; edges feed the PIT-filtered view-formation
    summary (visible from the NEXT decision day) and peer-concentration reporting."""
    import src.orchestrator as orch_mod
    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL", "BTC-USD", "SPY"]
    dates = pd.bdate_range("2015-01-01", periods=800).strftime("%Y-%m-%d").tolist()
    price_pivot = pd.DataFrame(100.0, index=dates, columns=["AAPL", "BTC-USD", "SPY"])
    monkeypatch.setattr(orch_mod, "get_pivot_close_prices", lambda *args, **kwargs: price_pivot)
    monkeypatch.setattr(orch_mod, "compute_barra_model", lambda p, t: (pd.DataFrame(), {}))
    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures", lambda *a, **k: np.zeros((3, 14)))
    synthetic_filing = {
        "cik": "0000320193", "filed": "2018-01-02",
        "accession_number": "0000320193-18-000001", "primary_document": "form10k.htm",
    }
    monkeypatch.setattr(orch_mod, "get_10k_filing_documents",
                        lambda ticker: [synthetic_filing] if ticker == "AAPL" else [])
    monkeypatch.setattr(orch_mod, "get_filing_text",
                        lambda filing: "Item 1 Business ... Item 1A Risk Factors ...")
    monkeypatch.setattr(orch_mod, "extract_relationships_from_filing",
                        lambda **kw: [("MSFT", "competitor")])

    captured = []
    def fake_view(**kwargs):
        captured.append(kwargs.get("relationships_summary"))
        return {"direction": "long", "conviction": 0.5, "rationale": "test"}
    monkeypatch.setattr(orch_mod, "call_view_formation_agent", fake_view)

    df_res = orch.run_daily_loop(start_date="2018-01-02", end_date="2018-01-04", is_warmup=True)
    assert len(orch.relationship_graph.edges) == 1
    assert orch.relationship_graph.edges[0]["relationship_type"] == "competitor"
    assert orch.relationship_graph.edges[0]["filed_date"] == "2018-01-02"

    # The filing-day summary is the explicit empty-set indicator (edge not yet usable).
    assert any(s.startswith("No company relationships on record") for s in captured)
    # From the next decision day onward the extracted relationship is visible to view formation.
    assert any("MSFT" in s and "competitor" in s for s in captured)

    # Peer-concentration diagnostic is reported at every rebalance.
    assert "Peer_Concentration" in df_res.columns


def test_orchestrator_uses_estimation_universe_for_barra_regression(monkeypatch):
    """Finding F2: the Barra factor regression runs on the separate, mechanically-selected
    ~300-name estimation universe — not the traded 3-asset book — and the two are distinct."""
    import src.orchestrator as orch_mod

    est_universe = [f"EST{i:04d}" for i in range(300)]
    monkeypatch.setattr(orch_mod, "build_estimation_universe", lambda size: est_universe)

    captured = {}
    def fake_compute_barra_model(price_pivot, estimation_tickers):
        captured["tickers"] = list(estimation_tickers)
        return pd.DataFrame(), {}
    monkeypatch.setattr(orch_mod, "compute_barra_model", fake_compute_barra_model)

    dates = pd.bdate_range("2015-01-01", periods=800).strftime("%Y-%m-%d").tolist()
    price_pivot = pd.DataFrame(100.0, index=dates,
                               columns=["AAPL", "BTC-USD", "SPY"] + est_universe)
    monkeypatch.setattr(orch_mod, "get_pivot_close_prices", lambda *args, **kwargs: price_pivot)

    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL", "BTC-USD", "SPY"]

    df_res = orch.run_daily_loop(start_date="2018-01-02", end_date="2018-01-02", is_warmup=True)
    assert not df_res.empty
    # The regression saw the mechanically-selected estimation universe, distinct from the traded book.
    assert captured.get("tickers") == est_universe
    assert len(captured["tickers"]) == 300
    assert set(captured["tickers"]).isdisjoint(set(orch.assets)) or set(captured["tickers"]) != set(orch.assets)
    assert orch.estimation_universe == est_universe


def test_daily_loop_invokes_cleanup(monkeypatch):
    """Finding F3: the daily memory-update step (Step 8.1) invokes the Step 2 cleanup rule for
    every asset on every decision day."""
    import src.orchestrator as orch_mod
    import src.memory.store as store_mod

    calls = []
    orig_cleanup = store_mod.AssetMemoryStore.run_cleanup
    def spy_cleanup(self):
        calls.append(self.asset_ticker)
        return orig_cleanup(self)
    monkeypatch.setattr(store_mod.AssetMemoryStore, "run_cleanup", spy_cleanup)

    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL", "BTC-USD", "SPY"]
    dates = pd.bdate_range("2015-01-01", periods=800).strftime("%Y-%m-%d").tolist()
    price_pivot = pd.DataFrame(100.0, index=dates, columns=["AAPL", "BTC-USD", "SPY"])
    monkeypatch.setattr(orch_mod, "get_pivot_close_prices", lambda *args, **kwargs: price_pivot)
    monkeypatch.setattr(orch_mod, "compute_barra_model", lambda p, t: (pd.DataFrame(), {}))
    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures",
                        lambda *a, **k: np.zeros((3, 14)))
    monkeypatch.setattr(orch_mod, "call_view_formation_agent",
                        lambda **kw: {"direction": "flat", "conviction": 0.0, "rationale": "test"})

    df_res = orch.run_daily_loop(start_date="2018-01-02", end_date="2018-01-05", is_warmup=True)
    days = len(df_res)
    # cleanup ran once per asset per decision day.
    assert len(calls) == 3 * days
    assert set(calls) == {"AAPL", "BTC-USD", "SPY"}


def test_daily_loop_cleanup_removes_qualifying_memories(monkeypatch):
    """Finding F3: over enough decision days a day-1 short memory crosses the recency threshold
    (delta >= 10 -> exp(-10/3) < 0.05) and the wired cleanup removes it; surviving short memories
    are all above both thresholds."""
    import src.orchestrator as orch_mod

    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL"]

    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2018-01-02", periods=16)]
    price_pivot = pd.DataFrame(100.0, index=dates, columns=["AAPL"])
    monkeypatch.setattr(orch_mod, "get_pivot_close_prices", lambda *args, **kwargs: price_pivot)
    monkeypatch.setattr(orch_mod, "compute_barra_model", lambda p, t: (pd.DataFrame(), {}))
    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures", lambda *a, **k: np.zeros((1, 14)))
    monkeypatch.setattr(orch_mod, "call_view_formation_agent",
                        lambda **kw: {"direction": "flat", "conviction": 0.0, "rationale": "test"})

    df_res = orch.run_daily_loop(start_date=dates[0], end_date=dates[-1], is_warmup=True)
    assert len(df_res) == len(dates)

    short = orch.stores["AAPL"].layers["short"]
    # One short memory is written per day (16), but those older than ~10 days are cleaned out.
    assert 0 < len(short) < 16
    for r in short:
        assert r.importance >= 5.0
        assert r.get_recency() >= 0.05


def test_step8_daily_steps_are_independently_callable(monkeypatch):
    """Structure Finding 2: the six extracted Step 8.1-8.6 methods can each be exercised directly
    (given their shared per-day state) without running the full daily loop end to end."""
    import src.orchestrator as orch_mod

    orch = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    orch.assets = ["AAPL"]

    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2015-01-01", periods=260)]
    price_pivot = pd.DataFrame(
        np.linspace(100.0, 150.0, len(dates)), index=dates, columns=["AAPL"]
    )
    day, prev_d = dates[258], dates[257]

    # Shared per-day state normally set by run_daily_loop.
    orch._price_pivot = price_pivot
    orch._all_dates = dates
    orch._style_stats = {}
    orch._barra_factor_map = {}
    orch.portfolio_value = 1.0
    orch._t_idx = 5
    orch._full_idx = 258
    orch._prev_d = prev_d

    monkeypatch.setattr(orch_mod, "get_point_in_time_fundamentals",
                        lambda a, decision_date=None, closing_price=None: {})
    monkeypatch.setattr(orch_mod, "call_view_formation_agent",
                        lambda **kw: {"direction": "long", "conviction": 0.5, "rationale": "t"})
    monkeypatch.setattr(orch_mod, "compute_asset_factor_exposures", lambda *a, **k: np.zeros((1, 14)))
    monkeypatch.setattr(orch_mod, "solve_portfolio_optimization", lambda **kw: np.array([0.5]))

    # Step 8.1: memory decay & migration + feedback.
    orch._decay_and_migrate_memory(day)

    # Step 8.2: view formation (and decision recording).
    views = orch._form_views(day)
    assert len(views) == 1
    assert views[0]["direction"] == "long"
    assert day in orch.feedback_tracker.decision_history

    # Step 8.3: portfolio construction + Step 6 share/value bookkeeping.
    w_t, p_ret_today = orch._construct_portfolio(day, views)
    assert w_t.shape == (1,)
    assert isinstance(p_ret_today, float)

    # Step 8.4: breadth / realized exposure / peer concentration reporting.
    breadth, realized_exp, peer_conc = orch._report_breadth_exposure(day, w_t)
    assert "Style_Size" in realized_exp

    # Step 8.5: reflection generation.
    orch._maybe_run_reflection(day)

    # Step 8.6: day-t short-layer memory write.
    orch._write_memory(day)

    # The evolving portfolio value was tracked by the Step 6 bookkeeping.
    assert orch.portfolio_value > 0
    # And each asset's short-layer day memory was written.
    assert len(orch.stores["AAPL"].layers["short"]) > 0
