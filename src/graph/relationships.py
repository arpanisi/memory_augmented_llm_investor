"""
Step 9 — Company Relationship Graph module.

Extracts company-to-company relationships (customers, suppliers, competitors) from SEC 10-K filing
text, validates them against a fixed closed candidate list, stores them as a directed multigraph
keyed by source filing `filed` date, and supports point-in-time querying.

Point-in-time rule: an edge extracted from a filing with `filed` date F only becomes usable starting
the NEXT decision day. For a decision day `t`, only edges with `filed < t` are visible (equivalently
`filed <= t - 1`). Edges are never deduplicated across filings: a relationship re-confirmed in a
later 10-K is a new, separate edge with its own `filed_date`.
"""
import json
import logging
import os

import requests

from src.agent.prompts import RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT
from config.settings import EDGAR_USER_AGENT

logger = logging.getLogger(__name__)

# The three allowed relationship types. Direction is encoded by which value is chosen:
#   - target_is_customer: target is the filer's customer.
#   - target_is_supplier: target is the filer's supplier.
#   - competitor:         symmetric (no direction).
RELATIONSHIP_TYPES = ("target_is_customer", "target_is_supplier", "competitor")


def build_candidate_tickers(traded_equities, estimation_universe) -> list[str]:
    """
    Builds the fixed closed list of valid target tickers for relationship extraction:
    the traded equities plus the factor-estimation-universe tickers. The source filer itself is
    rejected at validation time (a filer is not a relationship target of itself).
    """
    return sorted(set(traded_equities) | set(estimation_universe))


def format_relationship_extraction_user_prompt(
    source_ticker: str, filed_date: str, filing_text: str, candidate_tickers: list[str]
) -> str:
    """
    Formats the user prompt containing the filing text (Item 1 / Item 1A or first N chars of the
    body) plus the fixed closed list of valid target tickers.
    """
    prompt = f"Filer Ticker: {source_ticker}\nFiling Filed Date: {filed_date}\n\n"
    prompt += "--- Valid Target Tickers (use ONLY these) ---\n"
    prompt += ", ".join(candidate_tickers) + "\n\n"
    prompt += "--- Filing Text ---\n"
    prompt += filing_text + "\n"
    prompt += "\nReturn the JSON array of relationships now."
    return prompt


def parse_relationships_response(response_text: str) -> list[tuple[str, str]]:
    """
    Parses the raw LLM text into a list of (target_ticker, relationship_type) tuples.
    Returns raw, unvalidated tuples; validation happens in CompanyRelationshipGraph.
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    if not cleaned:
        return []

    try:
        data = json.loads(cleaned)
    except Exception:
        # Fall back to extracting a JSON array substring if extra prose is present
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(cleaned[start:end + 1])
        except Exception:
            return []

    if not isinstance(data, list):
        return []

    relationships = []
    for item in data:
        if not isinstance(item, dict):
            continue
        target = item.get("target_ticker")
        rtype = item.get("relationship_type")
        if target is None or rtype is None:
            continue
        relationships.append((str(target).strip().upper(), str(rtype).strip().lower()))
    return relationships


def extract_relationships_from_filing(
    source_ticker: str,
    filed_date: str,
    filing_text: str,
    candidate_tickers: list[str],
    model_id: str = "openai/gpt-4o-mini",
    api_key: str = None,
    use_fallback_if_offline: bool = True
) -> list[tuple[str, str]]:
    """
    Runs one LLM call against the filing text to extract raw (target_ticker, relationship_type)
    tuples. Returns the raw, unvalidated tuples ([] if the call fails or the text is empty).
    """
    if not filing_text or not filing_text.strip():
        logger.warning("Relationship extraction skipped for %s on %s: empty filing text.",
                       source_ticker, filed_date)
        return []

    api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    user_prompt = format_relationship_extraction_user_prompt(
        source_ticker, filed_date, filing_text, candidate_tickers
    )

    if not api_key or use_fallback_if_offline:
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 300
            }
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                raw_text = res.json()["choices"][0]["message"]["content"]
                return parse_relationships_response(raw_text)
        except Exception as e:
            logger.warning("Relationship extraction LLM call failed for %s: %s", source_ticker, e)

    # Deterministic offline degradation: no relationships can be asserted without a model call.
    logger.info("Relationship extraction fell back to empty result for %s on %s.", source_ticker, filed_date)
    return []


class CompanyRelationshipGraph:
    """
    Directed multigraph of company relationships.

    Nodes are tickers; edges are dicts {source, target, relationship_type, filed_date}. Edges are
    never deduplicated across filings.
    """

    def __init__(self, candidate_tickers: list[str] = None):
        self.candidate_tickers = set(candidate_tickers or [])
        self.edges: list[dict] = []

    def set_candidate_tickers(self, candidate_tickers: list[str]):
        self.candidate_tickers = set(candidate_tickers)

    def _validate(self, source: str, target: str, relationship_type: str) -> tuple[bool, str]:
        """Returns (ok, reason). Rejects unknown targets, self-targets, and invalid types."""
        if target not in self.candidate_tickers:
            return False, f"target_ticker '{target}' not in the fixed candidate list"
        if target == source:
            return False, f"target_ticker '{target}' is the source filer itself (self-reference)"
        if relationship_type not in RELATIONSHIP_TYPES:
            return False, f"relationship_type '{relationship_type}' is not one of {RELATIONSHIP_TYPES}"
        return True, ""

    def register_extraction_result(
        self, source: str, filed_date: str, relationships: list[tuple[str, str]]
    ) -> dict:
        """
        Validates a batch of (target, relationship_type) tuples from one extraction call, logs every
        rejection with a reason (never silently dropped), and stores the valid edges (no dedup).
        Returns {"added": [...], "rejected": [(target, type, reason), ...]}.
        """
        added = []
        rejected = []
        for target, relationship_type in relationships:
            target = str(target).strip().upper()
            relationship_type = str(relationship_type).strip().lower()
            ok, reason = self._validate(source, target, relationship_type)
            if not ok:
                logger.warning(
                    "Rejected relationship from %s (filed %s): target=%s type=%s reason=%s",
                    source, filed_date, target, relationship_type, reason
                )
                rejected.append((target, relationship_type, reason))
                continue
            edge = {
                "source": source,
                "target": target,
                "relationship_type": relationship_type,
                "filed_date": str(filed_date),
            }
            self.edges.append(edge)
            added.append(edge)
        return {"added": added, "rejected": rejected}

    def add_edge(self, source: str, target: str, relationship_type: str, filed_date: str) -> bool:
        """Directly adds a single validated edge. Returns True if stored, False if rejected."""
        result = self.register_extraction_result(source, filed_date, [(target, relationship_type)])
        return bool(result["added"])

    def visible_edges(self, decision_date: str) -> list[dict]:
        """
        Point-in-time filter: returns edges whose filing `filed` date is strictly before decision_date
        (i.e. filed <= decision_date - 1). Dates are ISO "%Y-%m-%d" strings, so lexicographic string
        comparison is chronologically correct.
        """
        return [e for e in self.edges if e["filed_date"] < str(decision_date)]

    def visible_edges_for_source(self, ticker: str, decision_date: str) -> list[dict]:
        return [e for e in self.visible_edges(decision_date) if e["source"] == ticker]

    def visible_competitor_edges(self, decision_date: str) -> list[dict]:
        return [e for e in self.visible_edges(decision_date) if e["relationship_type"] == "competitor"]

    def summarize_for_view(self, ticker: str, decision_date: str) -> str:
        """
        Summarizes the point-in-time-filtered outgoing edges of `ticker` for the daily view-formation
        LLM call, e.g. "named competitors: X, Y; named suppliers: Z". Returns an explicit empty-set
        indicator when there are no edges (never an omitted field).
        """
        edges = self.visible_edges_for_source(ticker, decision_date)
        if not edges:
            return "No company relationships on record (none extracted from any 10-K yet)."

        labels = {
            "competitor": "named competitors",
            "target_is_customer": "named customers",
            "target_is_supplier": "named suppliers",
        }
        parts = []
        for rtype in ("competitor", "target_is_customer", "target_is_supplier"):
            targets = sorted({e["target"] for e in edges if e["relationship_type"] == rtype})
            if targets:
                parts.append(f"{labels[rtype]}: {', '.join(targets)}")
        return "; ".join(parts)
