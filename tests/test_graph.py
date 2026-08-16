"""
Unit tests for Step 9 Company Relationship Graph (code-review Finding 1).
"""
import numpy as np
import pytest

from src.graph.relationships import (
    CompanyRelationshipGraph,
    build_candidate_tickers,
    parse_relationships_response,
)
from src.reporting.metrics import compute_peer_concentration_exposure


def test_build_candidate_tickers_closed_list():
    cands = build_candidate_tickers(["AAPL", "MSFT"], ["TSLA", "GOOGL"])
    assert set(cands) == {"AAPL", "MSFT", "TSLA", "GOOGL"}


def test_parse_relationships_response():
    raw = (
        "```json\n"
        '[{"target_ticker": "msft", "relationship_type": "competitor"},\n'
        ' {"target_ticker": "TSLA", "relationship_type": "target_is_customer"}]\n'
        "```"
    )
    rels = parse_relationships_response(raw)
    assert ("MSFT", "competitor") in rels
    assert ("TSLA", "target_is_customer") in rels


def test_parse_relationships_response_empty():
    assert parse_relationships_response("") == []
    assert parse_relationships_response("no relationships found") == []
    assert parse_relationships_response("[]") == []


def test_graph_rejects_invalid_target_and_type(caplog):
    g = CompanyRelationshipGraph(candidate_tickers=["AAPL", "MSFT", "TSLA"])
    res = g.register_extraction_result("AAPL", "2020-01-02", [
        ("MSFT", "target_is_customer"),   # valid
        ("TSLA", "competitor"),           # valid
        ("NOPE", "competitor"),           # invalid target ticker
        ("MSFT", "target_is_supplier"),   # valid
        ("AAPL", "competitor"),           # self-reference -> rejected
        ("MSFT", "weird_relationship"),   # invalid relationship_type
    ])
    assert len(res["added"]) == 3
    assert len(res["rejected"]) == 3
    reasons = " | ".join(r[2] for r in res["rejected"])
    assert "not in the fixed candidate list" in reasons
    assert "self-reference" in reasons
    assert "not one of" in reasons
    # Rejections are logged with a reason, never silently dropped.
    log_text = caplog.text.lower()
    assert "rejected relationship" in log_text


def test_graph_point_in_time_filtering():
    g = CompanyRelationshipGraph(candidate_tickers=["AAPL", "MSFT"])
    g.register_extraction_result("AAPL", "2020-01-02", [("MSFT", "competitor")])
    g.register_extraction_result("AAPL", "2020-01-05", [("MSFT", "target_is_supplier")])

    # On the filing date itself the edge is NOT visible (usable only from the NEXT decision day).
    assert len(g.visible_edges("2020-01-02")) == 0
    # One decision day after the first filing: only the first edge is visible.
    assert len(g.visible_edges("2020-01-03")) == 1
    # After the second filing, the following day sees both edges.
    assert len(g.visible_edges("2020-01-06")) == 2


def test_graph_no_dedup_across_filings():
    g = CompanyRelationshipGraph(candidate_tickers=["AAPL", "MSFT"])
    g.register_extraction_result("AAPL", "2020-01-02", [("MSFT", "competitor")])
    g.register_extraction_result("AAPL", "2021-01-02", [("MSFT", "competitor")])
    # A relationship re-confirmed in a later 10-K is a new, separate edge.
    assert len(g.edges) == 2
    assert {e["filed_date"] for e in g.edges} == {"2020-01-02", "2021-01-02"}


def test_graph_summarize_for_view_empty_and_nonempty():
    g = CompanyRelationshipGraph(candidate_tickers=["AAPL", "MSFT", "TSLA"])
    g.register_extraction_result("AAPL", "2020-01-02", [
        ("MSFT", "competitor"),
        ("TSLA", "target_is_supplier"),
    ])
    summary = g.summarize_for_view("AAPL", "2020-01-03")
    assert "competitors" in summary and "MSFT" in summary
    assert "suppliers" in summary and "TSLA" in summary

    # Empty-set indicator is explicit, never an omitted field.
    empty = g.summarize_for_view("MSFT", "2020-01-03")
    assert "No company relationships on record" in empty


def test_peer_concentration_counts_either_direction():
    w = np.array([0.3, 0.2, 0.1, 0.0])
    asset_names = ["AAA", "BBB", "CCC", "DDD"]
    traded = ["AAA", "BBB", "CCC"]
    edges = [
        {"source": "AAA", "target": "BBB", "relationship_type": "competitor", "filed_date": "2020-01-01"},
        {"source": "CCC", "target": "AAA", "relationship_type": "competitor", "filed_date": "2020-01-02"},
        {"source": "AAA", "target": "CCC", "relationship_type": "target_is_supplier", "filed_date": "2020-01-01"},
    ]
    res = compute_peer_concentration_exposure(w, asset_names, traded, edges)
    # AAA: peers BBB (0.2) + CCC (0.1), counting the incoming competitor edge from CCC too.
    assert res["AAA"] == pytest.approx(0.3)
    # BBB: connected to AAA (0.3).
    assert res["BBB"] == pytest.approx(0.3)
    # CCC: connected to AAA (0.3).
    assert res["CCC"] == pytest.approx(0.3)
