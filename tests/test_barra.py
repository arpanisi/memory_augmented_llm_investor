"""
Unit tests for the Barra factor model (code-review Findings 2, 4, 7).

The Barra pipeline is split across two modules (pure reorganization, code-review Structure
Finding 1): barra_estimation.py holds the offline daily cross-sectional regression
(compute_barra_model, compute_barra_daily_factors) and barra_exposures.py holds the per-traded-
asset exposure lookup (compute_asset_factor_exposures, _compute_daily_stock_row). The assertions
below are unchanged; only the module patch targets reflect the split.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import barra_estimation as est_mod
from src.data import barra_exposures as exp_mod

STYLE_KEYS = {"size", "value", "momentum", "volatility"}


def _synthetic_price_pivot(n_days=300, n_tickers=3, seed=1):
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    idx = [d.strftime("%Y-%m-%d") for d in dates]
    rng = np.random.RandomState(seed)
    prices = np.exp(np.cumsum(rng.randn(n_days, n_tickers) * 0.01, axis=0)) * 100.0
    tickers = ["AAA", "BBB", "CCC"][:n_tickers]
    return pd.DataFrame(prices, index=idx, columns=tickers)


def test_stock_row_excluded_when_fundamentals_missing(monkeypatch):
    """Finding 4: a missing shares/equity value excludes the stock from the regression; it is never
    replaced with a fabricated number."""
    def fake_fund(t, decision_date, closing_price=None):
        if t == "BBB":
            return {"shares_outstanding": None, "stockholders_equity": None, "sic": "2834"}
        return {"shares_outstanding": 1e9, "stockholders_equity": 1e12, "sic": "2834"}
    monkeypatch.setattr(exp_mod, "get_point_in_time_fundamentals", fake_fund)

    sic_map = {"AAA": "D", "BBB": "D"}
    row_aaa = exp_mod._compute_daily_stock_row(
        "AAA", 0.01, 100.0, pd.Series(dtype=float), pd.Series(dtype=float), sic_map, 0.0001, "2020-01-02"
    )
    row_bbb = exp_mod._compute_daily_stock_row(
        "BBB", 0.01, 100.0, pd.Series(dtype=float), pd.Series(dtype=float), sic_map, 0.0001, "2020-01-02"
    )
    assert row_aaa is not None
    assert row_bbb is None  # excluded rather than fabricated (old bug used 1e8 shares / 0.2*mcap)
    # The surviving stock's raw values come from its real fundamentals, not a fallback.
    assert row_aaa["mcap"] == pytest.approx(1e9 * 100.0)
    assert row_aaa["value_raw"] == pytest.approx(1e12 / 1e11)


def test_regression_runs_with_fewer_than_15_names(monkeypatch, tmp_path):
    """Finding 7: the invented <15 valid-names threshold is removed; a small cross-section still
    produces a full factor-return series (no silent gaps)."""
    price_pivot = _synthetic_price_pivot()
    idx = [d for d in price_pivot.index]

    monkeypatch.setattr(est_mod, "BARRA_CACHE_PATH", tmp_path / "barra.parquet")
    monkeypatch.setattr(est_mod, "fetch_risk_free_rate", lambda: pd.Series(0.0001, index=idx))

    # The estimation SIC map comes from the submissions index (get_sic_code), and the point-in-time
    # fundamentals panel is built once up front — both must be patched so this stays hermetic.
    monkeypatch.setattr(est_mod, "get_sic_code", lambda t: "2834")

    def fake_panel(tickers, dates):
        idx_grid = list(dates)
        return {
            t: pd.DataFrame(
                {"shares_outstanding": [1e9] * len(idx_grid),
                 "stockholders_equity": [1e12] * len(idx_grid)},
                index=idx_grid,
            )
            for t in tickers
        }
    monkeypatch.setattr(est_mod, "build_pit_fundamentals_panel", fake_panel)

    def fake_fund(t, decision_date, closing_price=None):
        return {"shares_outstanding": 1e9, "stockholders_equity": 1e12, "sic": "2834"}
    # _compute_daily_stock_row falls back to get_point_in_time_fundamentals when no panel is
    # supplied (the panel path is patched above), so it is patched too.
    monkeypatch.setattr(exp_mod, "get_point_in_time_fundamentals", fake_fund)

    df_factors, style_stats = est_mod.compute_barra_model(price_pivot, ["AAA", "BBB", "CCC"])
    # Only 3 names in the estimation universe: the old threshold would have produced an empty series.
    assert not df_factors.empty
    assert len(df_factors) > 30
    # Style stats (estimation-universe cross-sectional mean/std) are populated for every day.
    assert len(style_stats) == len(df_factors)
    assert STYLE_KEYS <= set(style_stats[list(style_stats)[0]].keys())


def test_asset_factor_exposures_structure_and_normalization(monkeypatch):
    """Finding 2: real (N, 14) exposure vectors are computed — industry dummy from the SIC division,
    style z-scores normalized against the estimation universe's cross-sectional stats (not the traded
    universe's own), and structural zero exposure for crypto/ETF assets."""
    price_pivot = _synthetic_price_pivot(n_days=300, n_tickers=2, seed=7)
    decision = price_pivot.index[260]

    # Estimation-universe cross-sectional stats: arbitrary mean/std so we can verify z-scoring uses
    # THESE stats rather than the traded universe's own.
    style_stats = {
        d: {
            "size": (5.0, 0.5),
            "value": (0.1, 0.02),
            "momentum": (0.0, 1.0),
            "volatility": (0.2, 0.05),
        }
        for d in price_pivot.index[252:]
    }

    def fake_fund(t, decision_date, closing_price=None):
        return {"shares_outstanding": 1e9, "stockholders_equity": 1e12, "sic": "4911"}
    monkeypatch.setattr(exp_mod, "get_point_in_time_fundamentals", fake_fund)
    monkeypatch.setattr(exp_mod, "get_sic_code", lambda t: "4911")  # Division E

    X = exp_mod.compute_asset_factor_exposures(
        price_pivot, ["AAA", "BBB", "BTC-USD", "SPY"], ["AAA", "BBB"], decision, style_stats
    )
    assert X is not None
    assert X.shape == (4, 14)

    # Crypto / ETF assets -> structural zero exposure to all 14 factors.
    assert np.all(X[2] == 0.0)
    assert np.all(X[3] == 0.0)

    # Industry dummy from SIC 4911 -> Division E (index 4 in A..J).
    assert X[0, 4] == pytest.approx(1.0)
    assert X[1, 4] == pytest.approx(1.0)

    # Size z-score normalized against the estimation-universe stats (mean 5.0, std 0.5).
    size_raw = np.log(1e9 * price_pivot.loc[decision, "AAA"])
    assert X[0, 10] == pytest.approx((size_raw - 5.0) / 0.5)

    # Value z-score uses estimation-universe value stats.
    value_raw = 1e12 / (1e9 * price_pivot.loc[decision, "AAA"])
    assert X[0, 11] == pytest.approx((value_raw - 0.1) / 0.02)


def test_barra_sic_map_from_submissions_not_companyfacts(monkeypatch, tmp_path):
    """Finding F1: the estimation-universe SIC divisions come from the EDGAR submissions index
    (get_sic_code), not the sic-less companyfacts cache, so the industry dummies are not uniformly
    Division D — names across different divisions produce non-zero factor-return columns for each."""
    price_pivot = _synthetic_price_pivot(n_days=300, n_tickers=3, seed=3)
    idx = [d for d in price_pivot.index]

    monkeypatch.setattr(est_mod, "BARRA_CACHE_PATH", tmp_path / "barra.parquet")
    monkeypatch.setattr(est_mod, "fetch_risk_free_rate", lambda: pd.Series(0.0001, index=idx))

    # Submissions-index SICs spanning three different divisions: E (utilities), D (manufacturing),
    # I (services). The old path read `facts["sic"]`, which is empty for every name and produced
    # Division D for all of them.
    sic_codes = {"AAA": "4911", "BBB": "2834", "CCC": "7372"}
    monkeypatch.setattr(est_mod, "get_sic_code", lambda t: sic_codes[t])

    def fake_panel(tickers, dates):
        idx_grid = list(dates)
        return {
            t: pd.DataFrame(
                {"shares_outstanding": [1e9] * len(idx_grid),
                 "stockholders_equity": [1e12] * len(idx_grid)},
                index=idx_grid,
            )
            for t in tickers
        }
    monkeypatch.setattr(est_mod, "build_pit_fundamentals_panel", fake_panel)

    df_factors, _ = est_mod.compute_barra_model(price_pivot, ["AAA", "BBB", "CCC"])
    assert not df_factors.empty
    # Each member's division shows a non-zero factor-return series (not all silently Division D).
    for ind_col in ["Ind_E", "Ind_D", "Ind_I"]:
        assert (df_factors[ind_col].abs() > 1e-12).sum() > 0
    # And more than one division is active across the series.
    active = [c for c in ["Ind_A", "Ind_B", "Ind_C", "Ind_D", "Ind_E", "Ind_F", "Ind_G", "Ind_H", "Ind_I", "Ind_J"]
              if (df_factors[c].abs() > 1e-12).sum() > 0]
    assert len(active) >= 3


def test_pit_fundamentals_sic_falls_back_to_submissions(monkeypatch):
    """Finding F1: companyfacts has no top-level 'sic', so get_point_in_time_fundamentals resolves
    the SIC from the same submissions-index source the traded-exposure path uses."""
    import src.data.edgar as edgar_mod

    def fake_facts(cik):
        return {"cik": cik, "entityName": "TEST", "facts": {"us-gaap": {
            "StockholdersEquity": {"units": {"USD": [
                {"end": "2017-12-31", "val": 1e12, "filed": "2017-11-03", "form": "10-K", "fy": 2017, "fp": "FY"}]}},
            "CommonStockSharesOutstanding": {"units": {"shares": [
                {"end": "2017-12-31", "val": 1e9, "filed": "2017-11-03", "form": "10-K", "fy": 2017, "fp": "FY"}]}},
        }}}
    monkeypatch.setattr(edgar_mod, "fetch_cik_mapping", lambda: {"TEST": "0000000001"})
    monkeypatch.setattr(edgar_mod, "fetch_company_facts", fake_facts)
    # No 'sic' anywhere in the facts payload -> must come from the submissions index.
    monkeypatch.setattr(edgar_mod, "get_sic_code", lambda t: "4911")

    fund = edgar_mod.get_point_in_time_fundamentals("TEST", decision_date="2018-01-02")
    assert fund["sic"] == "4911"
    assert fund["shares_outstanding"] == 1e9
