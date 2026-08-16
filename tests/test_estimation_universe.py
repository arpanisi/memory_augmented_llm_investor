"""
Unit tests for the mechanically-selected Factor Estimation Universe (code-review Finding F2).

The estimation universe is a separate ~300-name cross-section built by a mechanical rule (rank
the complete SEC registry by point-in-time market cap on the reference date, filtered by full
price history + PIT fundamentals) — it is never a hardcoded list. These tests exercise the
selection mechanism and cache discipline with a synthetic registry, so they are hermetic.
"""
import json

import pytest

from src.data import estimation_universe as eu_mod


def _build_hermetic(monkeypatch, tmp_path, size=2):
    pool = ["AAA", "BBB", "CCC", "DDD"]
    history = {"AAA": 10.0, "BBB": 50.0, "CCC": 20.0, "DDD": 5.0}
    shares = {"AAA": 1e9, "BBB": 1e8, "CCC": 5e8, "DDD": None}  # DDD: no PIT shares -> excluded

    def fake_fund(t, decision_date=None, closing_price=None):
        return {"shares_outstanding": shares[t], "stockholders_equity": 1e11, "sic": "4911"}

    monkeypatch.setattr(eu_mod, "_registry_tickers", lambda: pool)
    monkeypatch.setattr(eu_mod, "_names_with_full_history",
                        lambda pool_arg, ref, size_arg: (history, len(pool_arg)))
    monkeypatch.setattr(eu_mod, "get_point_in_time_fundamentals", fake_fund)
    cache_path = tmp_path / "estimation_universe.json"
    monkeypatch.setattr(eu_mod, "ESTIMATION_UNIVERSE_CACHE_PATH", cache_path)
    return cache_path


def test_selection_ranks_by_pit_market_cap_not_hardcoded(monkeypatch, tmp_path):
    """F2: names are ranked by reference-date price x point-in-time shares; a name with no PIT
    shares is excluded; the top-`size` by market cap are returned (not a curated list)."""
    cache_path = _build_hermetic(monkeypatch, tmp_path, size=2)

    selected = eu_mod.build_estimation_universe(size=2, force_refresh=True)
    # mcaps: AAA = 10 * 1e9 = 1e10; BBB = 50 * 1e8 = 5e9; CCC = 20 * 5e8 = 1e10; DDD excluded.
    assert selected == ["AAA", "CCC"]
    assert "DDD" not in selected
    # Distinct from any traded book by construction (synthetic here), but never hardcoded: the
    # rule output is what is returned.
    assert len(set(selected)) == 2


def test_selection_cache_roundtrip_and_method_marker(monkeypatch, tmp_path):
    """F2: the selection is cached to disk with a method/version marker so a stale or differently
    built cache is never silently reused."""
    cache_path = _build_hermetic(monkeypatch, tmp_path, size=2)

    first = eu_mod.build_estimation_universe(size=2, force_refresh=True)
    cached = json.loads(cache_path.read_text())
    assert cached["method"] == eu_mod.METHOD_NAME
    assert cached["size"] == 2
    assert cached["tickers"] == first

    # A matching cache short-circuits the selection (no re-ranking).
    second = eu_mod.build_estimation_universe(size=2, force_refresh=False)
    assert second == first


def test_mismatched_cache_is_rebuilt(monkeypatch, tmp_path):
    """F2: a cache with a different size/method/ref-date is ignored and rebuilt, not reused."""
    cache_path = _build_hermetic(monkeypatch, tmp_path, size=2)
    eu_mod.build_estimation_universe(size=2, force_refresh=True)

    # A stale cache for a different size must not be returned for size=3.
    result = eu_mod.build_estimation_universe(size=3, force_refresh=False)
    assert len(result) == 3
    assert json.loads(cache_path.read_text())["size"] == 3


def test_selection_fails_if_too_few_qualify(monkeypatch, tmp_path):
    """F2: a mechanical build that cannot assemble `size` qualifying names raises instead of
    silently returning a short universe."""
    cache_path = _build_hermetic(monkeypatch, tmp_path, size=2)
    monkeypatch.setattr(eu_mod, "_names_with_full_history",
                        lambda pool, ref, size: ({"AAA": 1.0}, 4))
    monkeypatch.setattr(eu_mod, "get_point_in_time_fundamentals",
                        lambda t, decision_date=None, closing_price=None:
                        {"shares_outstanding": 1e9, "stockholders_equity": 1e11})
    with pytest.raises(RuntimeError):
        eu_mod.build_estimation_universe(size=5, force_refresh=True)
