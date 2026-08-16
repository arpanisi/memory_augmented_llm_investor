"""
Prompt template and formatting utilities for asset view formation.
Enforces structured JSON output schema and inclusion of explicit fundamentals sentinel for Crypto/ETFs.
"""
import json

VIEW_FORMATION_SYSTEM_PROMPT = """
You are an expert quantitative trading agent tasked with analyzing an asset's price history, point-in-time fundamentals, and retrieved past memories to form a trading decision.

You MUST respond ONLY with a valid JSON object matching this schema:
{
  "direction": "long" | "short" | "flat",
  "conviction": float between 0.0 and 1.0,
  "rationale": "1-2 sentences citing specific evidence (memory, fundamental figure, price pattern)."
}

Rules:
- direction must be exactly one of "long", "short", or "flat".
- conviction must be a float in [0.0, 1.0] representing confidence.
- rationale must cite specific evidence from the inputs.
- Do not output any markdown code blocks or extra text outside the JSON object.
"""

RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT = """
You are an expert at analyzing SEC 10-K filing text. Extract company-to-company business relationships that are explicitly stated in the filing.

You MUST respond ONLY with a valid JSON array. Each element is an object with exactly two fields:
{"target_ticker": "TICKER", "relationship_type": "TYPE"}

relationship_type must be exactly one of:
- "target_is_customer": the target company is the filer's customer.
- "target_is_supplier": the target company is the filer's supplier.
- "competitor": the target company is a competitor of the filer (symmetric, no direction).

Rules:
- Use ONLY target tickers from the provided "Valid Target Tickers" list.
- There is no separate direction field; direction is encoded by which relationship_type you choose.
- Only include relationships that are explicitly supported by the filing text.
- Return [] if no relationships are found.
- Do not output any markdown code blocks or extra text outside the JSON array.
"""

def format_view_formation_user_prompt(
    asset_ticker: str,
    decision_date: str,
    price_history_20d: list[dict],
    fundamentals: dict,
    retrieved_memories: list[dict],
    relationships_summary: str = "No company relationships on record."
) -> str:
    """
    Formats the user prompt containing 20-day prices, fundamentals (or sentinel), retrieved memories,
    and the point-in-time company-relationship summary.
    """
    prompt = f"Asset: {asset_ticker}\nDecision Date: {decision_date}\n\n"
    
    # 1. Price History (Trailing 20 days through decision_date - 1)
    prompt += "--- Trailing 20-Day Price History ---\n"
    for p in price_history_20d:
        prompt += f"Date: {p['Date']}, Close: {p['Close']:.2f}, Return: {p.get('Return', 0.0):+.2%}\n"
    
    # 2. Point-in-Time Fundamentals
    prompt += "\n--- Point-in-Time Fundamentals ---\n"
    if not fundamentals.get("fundamentals_available", True):
        prompt += f"fundamentals_available: false ({fundamentals.get('reason', 'No SEC filing available')})\n"
    else:
        prompt += f"P/E Ratio: {fundamentals.get('pe_ratio')}\n"
        prompt += f"P/B Ratio: {fundamentals.get('pb_ratio')}\n"
        prompt += f"Net Income: {fundamentals.get('net_income')}\n"
        prompt += f"Equity: {fundamentals.get('stockholders_equity')}\n"

    # 3. Company Relationships (point-in-time filtered, Step 9)
    prompt += "\n--- Company Relationships ---\n"
    prompt += relationships_summary + "\n"

    # 4. Retrieved Memories (Up to 20 total across short, mid, long, reflection)
    prompt += "\n--- Retrieved Memories ---\n"
    if not retrieved_memories:
        prompt += "No past memories retrieved.\n"
    else:
        for idx, m in enumerate(retrieved_memories, 1):
            prompt += f"[{idx}] Layer: {m['layer']}, Date: {m['date_written']}, Text: {m['text']} (Importance: {m['importance']:.1f}, Recency: {m['recency']:.2f})\n"

    prompt += "\nFormulate your trading view in JSON now."
    return prompt
