"""
Global settings and hyperparameter specifications locked by coding-plan.md.
"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# --- Candidate Pool & Fixed Books ---
CANDIDATE_POOL = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "JNJ", "JPM", "V", "PG", "UNH",
    "HD", "MA", "DIS", "BAC", "XOM", "CVX", "KO", "PEP", "WMT", "MRK",
    "ABBV", "PFE", "CSCO", "ADBE", "CRM", "NFLX", "INTC", "VZ", "T", "CMCSA",
    "NKE", "MCD", "COST", "TXN", "HON", "UPS", "IBM", "GE", "CAT", "BA"
]

CRYPTO_BOOK = ["BTC-USD", "ETH-USD"]
ETF_BOOK = ["SPY", "QQQ", "GLD"]

EQUITIES_BOOK_SIZE = 10
FACTOR_ESTIMATION_UNIVERSE_SIZE = 300

# --- Date Boundaries ---
PRICE_START_DATE = "2015-01-01"
PRICE_END_DATE = "2023-12-31"
SELECTION_REF_DATE = "2018-01-01"

WARMUP_START_DATE = "2018-01-01"
WARMUP_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2023-12-31"

# --- SIC Division Mapping ---
# Division A (01-09), B (10-14), C (15-17), D (20-39), E (40-49), F (50-51), G (52-59), H (60-67), I (70-89), J (91-99)
SIC_DIVISIONS = {
    "A": list(range(1, 10)),
    "B": list(range(10, 15)),
    "C": list(range(15, 18)),
    "D": list(range(20, 40)),
    "E": list(range(40, 50)),
    "F": list(range(50, 52)),
    "G": list(range(52, 60)),
    "H": list(range(60, 68)),
    "I": list(range(70, 90)),
    "J": list(range(91, 100)),
}

# --- Memory System Constants ---
MEMORY_LAYERS = {
    "short": {"initial_importance": 50.0, "Q": 3.0, "decay": 0.92},
    "mid": {"initial_importance": 60.0, "Q": 90.0, "decay": 0.96},
    "long": {"initial_importance": 90.0, "Q": 365.0, "decay": 0.96},
    "reflection": {"initial_importance": 80.0, "Q": 365.0, "decay": 0.98},
}

MIGRATION_SHORT_MID = 55.0
MIGRATION_MID_LONG = 85.0

CLEANUP_IMPORTANCE_MIN = 5.0
CLEANUP_RECENCY_MIN = 0.05

RETRIEVAL_TOP_K_PER_LAYER = 5

FEEDBACK_LOOKBACK_DAYS = 5
FEEDBACK_DELTA_IMPORTANCE = 18.0

REFLECTION_INTERVAL_DAYS = 5
REFLECTION_DEDUP_SIMILARITY = 0.95

# --- Portfolio Optimization Constants ---
LAMBDA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
COVARIANCE_WINDOW_DAYS = 252

INDUSTRY_FACTOR_BOUND = 0.5
STYLE_FACTOR_BOUND = 0.3

# --- SEC EDGAR Config ---
EDGAR_USER_AGENT = "QuantProjectsResearch investor-agent@quantprojects.org"

# --- Company Relationship Graph (Step 9) ---
# Available estimation-universe ticker pool used as the closed candidate list for relationship
# extraction (the locked 300-name universe is not separately materialized in this codebase, so the
# 40-name candidate pool is the available ticker set). The other 9 traded equities are added per-source.
FACTOR_ESTIMATION_UNIVERSE = CANDIDATE_POOL

# Maximum characters of filing text (Item 1 / Item 1A, or the filing body) sent to the
# relationship-extraction LLM call.
FILING_TEXT_MAX_CHARS = 15000
