"""
Tier 1 — Smoke Test Script.
Runs 3 assets (1 equity, BTC-USD, SPY) over 10 trading days drawn from the warmup window.
Confirms end-to-end execution of data, memory, view formation, optimization, reporting, and daily loop.
"""
import sys
from pathlib import Path

# Ensure src and config are in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.orchestrator import DailyOrchestrator
from src.evaluation.metrics import compute_evaluation_summary

def main():
    print("=== Launching Tier 1 Smoke Test ===")
    orchestrator = DailyOrchestrator(model_id="openai/gpt-4o-mini", risk_aversion_lambda=1.0)
    
    # Restrict to 3 smoke assets
    orchestrator.assets = ["AAPL", "BTC-USD", "SPY"]
    
    # Run 10 trading days in January 2018
    df_results = orchestrator.run_daily_loop(start_date="2018-01-02", end_date="2018-01-16", is_warmup=True)
    
    print("\n--- Tier 1 Execution Results ---")
    print(df_results.to_string())
    
    if not df_results.empty and "Portfolio_Return" in df_results.columns:
        summary = compute_evaluation_summary(df_results["Portfolio_Return"], model_name="Tier1_Smoke")
        print("\n--- Evaluation Summary ---")
        for k, v in summary.items():
            print(f"{k}: {v}")
            
    print("\nTier 1 Smoke Test completed successfully!")

if __name__ == "__main__":
    main()
