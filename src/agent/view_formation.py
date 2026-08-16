"""
LLM View Formation module supporting OpenRouter closed-source models and self-hosted vLLM open-weight models.
Outputs structured (direction, conviction, rationale) triples per asset.
"""
import os
import json
import requests
from src.agent.prompts import VIEW_FORMATION_SYSTEM_PROMPT, format_view_formation_user_prompt

def parse_view_formation_response(response_text: str) -> dict:
    """
    Parses raw LLM text into validated direction, conviction, rationale.
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        direction = str(data.get("direction", "flat")).lower()
        if direction not in ["long", "short", "flat"]:
            direction = "flat"
        
        conviction = float(data.get("conviction", 0.5))
        conviction = max(0.0, min(1.0, conviction))
        
        rationale = str(data.get("rationale", "Standard evaluation based on inputs."))
        return {
            "direction": direction,
            "conviction": conviction,
            "rationale": rationale
        }
    except Exception:
        # Fallback if parsing fails
        return {
            "direction": "flat",
            "conviction": 0.0,
            "rationale": "Fallback view due to response parsing exception."
        }

def generate_heuristic_fallback_view(price_history_20d: list[dict], fundamentals: dict) -> dict:
    """
    Deterministic view formation fallback when running offline or without API connectivity.
    """
    if not price_history_20d:
        return {"direction": "flat", "conviction": 0.0, "rationale": "Insufficient price history."}

    recent_ret = (price_history_20d[-1]["Close"] / price_history_20d[0]["Close"]) - 1.0 if len(price_history_20d) > 1 else 0.0
    
    if recent_ret > 0.02:
        return {"direction": "long", "conviction": min(0.9, 0.5 + recent_ret * 2), "rationale": f"Positive 20-day momentum ({recent_ret:+.2%})."}
    elif recent_ret < -0.02:
        return {"direction": "short", "conviction": min(0.9, 0.5 + abs(recent_ret) * 2), "rationale": f"Negative 20-day momentum ({recent_ret:+.2%})."}
    else:
        return {"direction": "flat", "conviction": 0.3, "rationale": f"Neutral 20-day price trend ({recent_ret:+.2%})."}

def call_view_formation_agent(
    asset_ticker: str,
    decision_date: str,
    price_history_20d: list[dict],
    fundamentals: dict,
    retrieved_memories: list[dict],
    relationships_summary: str = "No company relationships on record.",
    model_id: str = "openai/gpt-4o-mini",
    api_key: str = None,
    use_fallback_if_offline: bool = True
) -> dict:
    """
    Executes a single view-formation decision for asset_ticker on decision_date.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    user_prompt = format_view_formation_user_prompt(
        asset_ticker, decision_date, price_history_20d, fundamentals, retrieved_memories,
        relationships_summary=relationships_summary
    )

    if not api_key or use_fallback_if_offline:
        # Check if live HTTP call is possible, else return deterministic fallback
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": VIEW_FORMATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 200
            }
            res = requests.post(url, headers=headers, json=payload, timeout=10)
            if res.status_code == 200:
                raw_text = res.json()["choices"][0]["message"]["content"]
                return parse_view_formation_response(raw_text)
        except Exception:
            pass

    return generate_heuristic_fallback_view(price_history_20d, fundamentals)
