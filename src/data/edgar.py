"""
SEC EDGAR REST API integration for point-in-time fundamentals and filing text.
Enforces structual point-in-time correctness via the EDGAR `filed` date.
"""
import json
import re
import time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from config.settings import DATA_DIR, EDGAR_USER_AGENT, CRYPTO_BOOK, ETF_BOOK, FILING_TEXT_MAX_CHARS

EDGAR_CACHE_DIR = DATA_DIR / "edgar_cache"
FILING_TEXT_CACHE_DIR = EDGAR_CACHE_DIR / "filing_text"
TICKERS_JSON_PATH = EDGAR_CACHE_DIR / "company_tickers.json"

# In-memory caches so repeated point-in-time lookups (one per estimation-universe name per
# regression day) do not re-read the ~5MB facts / submissions JSON from disk every call.
_cik_map_cache: dict[str, str] | None = None
_facts_cache: dict[str, dict] = {}
_submissions_cache: dict[str, dict] = {}

def _get_headers() -> dict:
    return {"User-Agent": EDGAR_USER_AGENT}

def fetch_cik_mapping(force_refresh: bool = False) -> dict[str, str]:
    """
    Downloads SEC company_tickers.json and returns a dict mapping ticker -> 10-digit padded CIK string.
    Result is memoized in-process; pass force_refresh=True to re-download.
    """
    global _cik_map_cache
    if _cik_map_cache is not None and not force_refresh:
        return _cik_map_cache
    EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if TICKERS_JSON_PATH.exists() and not force_refresh:
        with open(TICKERS_JSON_PATH, "r") as f:
            data = json.load(f)
    else:
        url = "https://www.sec.gov/files/company_tickers.json"
        res = requests.get(url, headers=_get_headers(), timeout=15)
        res.raise_for_status()
        data = res.json()
        with open(TICKERS_JSON_PATH, "w") as f:
            json.dump(data, f)

    mapping = {}
    for entry in data.values():
        t = str(entry["ticker"]).upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[t] = cik
    _cik_map_cache = mapping
    return mapping

def fetch_company_facts(cik: str, force_refresh: bool = False) -> dict:
    """
    Fetches raw companyfacts JSON for a given 10-digit CIK. Memoized in-process; pass
    force_refresh=True to re-download from EDGAR.
    """
    if not force_refresh and cik in _facts_cache:
        return _facts_cache[cik]
    EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = EDGAR_CACHE_DIR / f"facts_{cik}.json"
    if file_path.exists() and not force_refresh:
        with open(file_path, "r") as f:
            facts = json.load(f)
        _facts_cache[cik] = facts
        return facts

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    time.sleep(0.12)  # Rate limit safety (<10 req/sec)
    try:
        res = requests.get(url, headers=_get_headers(), timeout=15)
        if res.status_code == 200:
            facts = res.json()
            with open(file_path, "w") as f:
                json.dump(facts, f)
            _facts_cache[cik] = facts
            return facts
    except Exception as e:
        print(f"Warning: Failed to fetch SEC facts for CIK {cik}: {e}")
    return {}

# Concepts holding shares-outstanding counts, plus plausibility guards: EDGAR filers occasionally
# report share counts in thousands/millions (values 1e3x-1e6x too large), which would otherwise
# fabricate a market cap. Two mechanical guards drop such errors:
#   - SHARES_MIN/MAX: an absolute band. No US-listed company has fewer than ~100k or more than
#     ~100B shares outstanding; values outside it (e.g. a filer reporting in units of thousands
#     for its whole history) are rejected outright.
#   - SHARES_JUMP_FILTER: a relative guard. Values that jump >50x from the previously-accepted
#     value (no legitimate split is anywhere near 50x) are rejected, carrying the last plausible
#     count forward point-in-time.
SHARES_OUTSTANDING_CONCEPTS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]
SHARES_MIN = 1e5
SHARES_MAX = 1e11
SHARES_JUMP_FILTER = 50.0

def _plausible_shares_series(df_facts: pd.DataFrame) -> pd.Series:
    """
    Returns a filed-date-sorted Series of plausibility-filtered shares-outstanding values
    (latest-filed value per date wins). Values outside the absolute band or jumping >50x from the
    prior accepted value are dropped; the last plausible count is carried forward point-in-time.
    """
    sub = df_facts[df_facts["concept"].isin(SHARES_OUTSTANDING_CONCEPTS)]
    if sub.empty:
        return pd.Series(dtype=float)
    sub = sub.sort_values("filed", kind="stable")
    kept_rows = []
    last_kept = None
    for _, row in sub.iterrows():
        val = row["val"]
        if val is None or pd.isna(val):
            continue
        val = float(val)
        if val < SHARES_MIN or val > SHARES_MAX:
            continue
        if last_kept is None:
            kept_rows.append(row)
            last_kept = val
            continue
        ratio = val / last_kept if last_kept > 0 else 0.0
        if ratio <= SHARES_JUMP_FILTER and 1.0 / ratio <= SHARES_JUMP_FILTER:
            kept_rows.append(row)
            last_kept = val
    if not kept_rows:
        return pd.Series(dtype=float)
    out = pd.DataFrame(kept_rows).drop_duplicates(subset="filed", keep="last")
    return pd.Series(out["val"].astype(float).to_numpy(), index=out["filed"].to_numpy(), name="shares_outstanding")


def extract_fact_time_series(facts_json: dict, concept_names: list[str], taxonomy: str = "us-gaap") -> pd.DataFrame:
    """
    Extracts facts for specified XBRL concept names into a pandas DataFrame with columns:
    [concept, form, filed, frame, val, fy, fp].
    """
    records = []
    facts_data = facts_json.get("facts", {}).get(taxonomy, {})
    for concept in concept_names:
        if concept in facts_data:
            units = facts_data[concept].get("units", {})
            for unit_key, fact_list in units.items():
                for f in fact_list:
                    if "filed" in f and "val" in f:
                        records.append({
                            "concept": concept,
                            "form": f.get("form", ""),
                            "filed": f["filed"],
                            "val": f["val"],
                            "fy": f.get("fy"),
                            "fp": f.get("fp")
                        })
    if not records:
        return pd.DataFrame(columns=["concept", "form", "filed", "val", "fy", "fp"])
    df = pd.DataFrame(records)
    df["filed"] = pd.to_datetime(df["filed"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("filed")

def get_point_in_time_fundamentals(ticker: str, decision_date: str, closing_price: float = None) -> dict:
    """
    Retrieves the most recent point-in-time fundamentals for ticker as of decision_date (filed <= decision_date).
    Returns sentinel dictionary for Crypto and ETF assets.
    """
    if ticker in CRYPTO_BOOK or ticker in ETF_BOOK:
        return {
            "fundamentals_available": False,
            "ticker": ticker,
            "decision_date": decision_date,
            "reason": "Crypto and ETF assets do not have EDGAR filings."
        }

    cik_map = fetch_cik_mapping()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return {
            "fundamentals_available": False,
            "ticker": ticker,
            "decision_date": decision_date,
            "reason": f"No SEC CIK found for ticker {ticker}"
        }

    facts = fetch_company_facts(cik)
    if not facts:
        return {
            "fundamentals_available": False,
            "ticker": ticker,
            "decision_date": decision_date,
            "reason": f"No EDGAR facts returned for CIK {cik}"
        }

    # Extract target XBRL tags
    concepts = [
        "NetIncomeLoss", "StockholdersEquity", "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding", "Revenues", "Assets", "Liabilities"
    ]
    df_facts = extract_fact_time_series(facts, concepts)
    
    # Filter strictly by point-in-time filed date
    pit_facts = df_facts[df_facts["filed"] <= decision_date]
    
    if pit_facts.empty:
        return {
            "fundamentals_available": False,
            "ticker": ticker,
            "decision_date": decision_date,
            "reason": f"No filings filed on or before {decision_date}"
        }

    latest_vals = {}
    for concept in concepts:
        sub = pit_facts[pit_facts["concept"] == concept]
        if not sub.empty:
            latest_vals[concept] = sub.iloc[-1]["val"]
        else:
            latest_vals[concept] = None

    # Shares outstanding: take the latest plausibility-filtered count as of the decision date
    # (a bad filing that jumps >50x from the prior count is dropped, never fabricated).
    shares = None
    shares_series = _plausible_shares_series(df_facts)
    if not shares_series.empty:
        valid_shares = shares_series[shares_series.index <= decision_date]
        if not valid_shares.empty:
            shares = float(valid_shares.iloc[-1])
    net_income = latest_vals.get("NetIncomeLoss")
    equity = latest_vals.get("StockholdersEquity")
    
    # Calculate P/E and P/B if price is provided and values exist
    pe_ratio = None
    pb_ratio = None
    if closing_price and shares and shares > 0:
        if net_income is not None:
            eps = net_income / shares
            if eps != 0:
                pe_ratio = float(closing_price / eps)
        if equity is not None:
            bps = equity / shares
            if bps != 0:
                pb_ratio = float(closing_price / bps)

    # companyfacts carries no top-level "sic" field, so resolve the SIC from the EDGAR
    # submissions index (the same source the traded-exposure path uses via get_sic_code).
    sic = str(facts.get("sic", "") or "")
    if not sic:
        sic = get_sic_code(ticker)

    return {
        "fundamentals_available": True,
        "ticker": ticker,
        "cik": cik,
        "sic": sic,
        "decision_date": decision_date,
        "net_income": net_income,
        "stockholders_equity": equity,
        "shares_outstanding": shares,
        "revenues": latest_vals.get("Revenues"),
        "pe_ratio": pe_ratio,
        "pb_ratio": pb_ratio,
    }

def fetch_company_submissions(cik: str, force_refresh: bool = False) -> dict:
    """
    Fetches the EDGAR submissions index (companyfacts does not carry SIC codes or filing documents).
    Cached to disk and memoized in-process. Returns {} on failure.
    """
    if not force_refresh and cik in _submissions_cache:
        return _submissions_cache[cik]
    EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = EDGAR_CACHE_DIR / f"submissions_{cik}.json"
    if file_path.exists() and not force_refresh:
        with open(file_path, "r") as f:
            data = json.load(f)
        _submissions_cache[cik] = data
        return data

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(0.12)
    try:
        res = requests.get(url, headers=_get_headers(), timeout=15)
        if res.status_code == 200:
            data = res.json()
            with open(file_path, "w") as f:
                json.dump(data, f)
            _submissions_cache[cik] = data
            return data
    except Exception as e:
        print(f"Warning: Failed to fetch SEC submissions for CIK {cik}: {e}")
    return {}

def get_sic_code(ticker: str) -> str:
    """
    Returns the SIC code for ticker from the EDGAR submissions index (cached), or '' if unavailable.
    Used to look up the day's SIC division for the 10 industry factor dummies.
    """
    if ticker in CRYPTO_BOOK or ticker in ETF_BOOK:
        return ""
    cik_map = fetch_cik_mapping()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return ""
    subs = fetch_company_submissions(cik)
    return str(subs.get("sic", ""))


def build_pit_fundamentals_panel(tickers: list[str], dates) -> dict[str, pd.DataFrame]:
    """
    Efficiently builds a point-in-time fundamentals panel for a set of tickers over a date grid.

    The estimation-universe regression needs latest-`filed`-date shares/equity for every name on
    every day; calling get_point_in_time_fundamentals per (name, date) would re-parse the ~5MB
    companyfacts JSON hundreds of thousands of times. Instead the XBRL concept series for each
    name is extracted once and pd.merge_asof carries forward the most recent value filed on or
    before each grid date.

    Returns {ticker: DataFrame(index=grid dates as "%Y-%m-%d" strings,
                               columns=["shares_outstanding", "stockholders_equity"])}
    where a missing value is NaN (never a fabricated number); the regression excludes the name
    on a day its value is unavailable.
    """
    cik_map = fetch_cik_mapping()
    grid = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    grid_index = [d.strftime("%Y-%m-%d") for d in grid]
    shares_concepts = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]
    equity_concepts = ["StockholdersEquity"]

    def _merge_concept_asof(concept_names: list[str], col_name: str) -> pd.Series:
        if df_facts.empty:
            return pd.Series(np.nan, index=grid_index, name=col_name)
        sub = df_facts[df_facts["concept"].isin(concept_names)]
        if sub.empty:
            return pd.Series(np.nan, index=grid_index, name=col_name)
        sub = sub.copy()
        sub["_d"] = pd.to_datetime(sub["filed"])
        # df_facts is already sorted by `filed`; keep a stable order so that records sharing a
        # filed date resolve to the same last value get_point_in_time_fundamentals returns.
        sub = sub.sort_values("_d", kind="stable").drop_duplicates(subset="_d", keep="last")
        merged = pd.merge_asof(
            pd.DataFrame({"date": grid}),
            sub[["_d", "val"]],
            left_on="date",
            right_on="_d",
            direction="backward",
        )
        return pd.Series(merged["val"].to_numpy(), index=grid_index, name=col_name)

    panel = {}
    for t in tickers:
        cik = cik_map.get(t.upper())
        df_facts = pd.DataFrame()
        if cik:
            facts = fetch_company_facts(cik)
            if facts:
                df_facts = extract_fact_time_series(facts, shares_concepts + equity_concepts)

        # Shares: plausibility-filtered series carried forward point-in-time onto the grid via
        # merge_asof (handles values filed before the grid starts, unlike a plain grid ffill).
        shares_series = _plausible_shares_series(df_facts)
        shares = pd.Series(np.nan, index=grid_index, name="shares_outstanding")
        if not shares_series.empty:
            right = pd.DataFrame({
                "filed_d": pd.to_datetime(pd.Series(shares_series.index)),
                "val": shares_series.to_numpy(),
            }).sort_values("filed_d")
            merged = pd.merge_asof(
                pd.DataFrame({"date": grid}),
                right,
                left_on="date",
                right_on="filed_d",
                direction="backward",
            )
            shares = pd.Series(merged["val"].to_numpy(), index=grid_index)

        equity = _merge_concept_asof(equity_concepts, "equity")

        panel[t] = pd.DataFrame({
            "shares_outstanding": shares.to_numpy(),
            "stockholders_equity": equity.to_numpy(),
        }, index=grid_index)

    return panel

def get_10k_filing_documents(ticker: str) -> list[dict]:
    """
    Returns 10-K filings for ticker as a list of dicts:
    {cik, filed, accession_number, primary_document}. Uses the EDGAR submissions index.
    Returns [] if unavailable.
    """
    if ticker in CRYPTO_BOOK or ticker in ETF_BOOK:
        return []
    cik_map = fetch_cik_mapping()
    cik = cik_map.get(ticker.upper())
    if not cik:
        return []
    subs = fetch_company_submissions(cik)
    recent = subs.get("recent") or {}
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    primaries = recent.get("primaryDocument") or []

    docs = []
    for i in range(len(forms)):
        if forms[i] != "10-K":
            continue
        accession = accessions[i] if i < len(accessions) else ""
        primary = primaries[i] if i < len(primaries) else ""
        if not accession or not primary:
            continue
        docs.append({
            "cik": cik,
            "filed": str(filing_dates[i]) if i < len(filing_dates) else "",
            "accession_number": str(accession),
            "primary_document": str(primary),
        })
    return docs

def extract_item_sections(raw_text: str, max_chars: int = FILING_TEXT_MAX_CHARS) -> str:
    """
    Isolates the 'Item 1' (Business) and 'Item 1A' (Risk Factors) section text from a 10-K.
    If either section cannot be isolated, returns the first max_chars characters of the body.
    """
    text = re.sub(r"<[^>]+>", " ", raw_text)  # strip HTML tags
    text = re.sub(r"\s+", " ", text)

    pat_item1 = re.search(r"Item\s*1\.?\s*Business", text, re.IGNORECASE)
    pat_1a = re.search(r"Item\s*1A\.?\s*Risk\s*Factors", text, re.IGNORECASE)

    if pat_item1 and pat_1a:
        start = pat_item1.start()
        end = pat_1a.start() + len(pat_1a.group(0))
        return text[start:end][:max_chars]
    if pat_item1:
        return text[pat_item1.start():][:max_chars]
    return text[:max_chars]

def get_filing_text(filing: dict, max_chars: int = FILING_TEXT_MAX_CHARS) -> str:
    """
    Downloads and returns the primary 10-K document text for a filing dict (as returned by
    get_10k_filing_documents), truncated to max_chars after section isolation.
    Returns "" (and logs) if the document cannot be retrieved.
    """
    cik = filing.get("cik")
    accession = str(filing.get("accession_number", "")).replace("-", "")
    primary = filing.get("primary_document")
    if not cik or not accession or not primary:
        return ""

    FILING_TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FILING_TEXT_CACHE_DIR / f"{cik}_{accession}.txt"
    if cache_path.exists():
        raw = cache_path.read_text(encoding="utf-8", errors="ignore")
        return extract_item_sections(raw, max_chars)

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary}"
    time.sleep(0.12)
    try:
        res = requests.get(url, headers=_get_headers(), timeout=20)
        res.raise_for_status()
        raw = res.text
        cache_path.write_text(raw, encoding="utf-8")
        return extract_item_sections(raw, max_chars)
    except Exception as e:
        print(f"Warning: Failed to fetch 10-K text for CIK {cik} accession {accession}: {e}")
        return ""
