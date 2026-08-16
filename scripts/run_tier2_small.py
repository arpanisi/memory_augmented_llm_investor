"""
Tier 2 — Small Test Script.
Full 15-asset universe, 3 months of warmup + 1 month of test, lambda grid search, and baselines comparison.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.orchestrator import DailyOrchestrator, run_lambda_grid_search
from src.data.prices import get_pivot_close_prices
from src.evaluation.baselines import run_buy_and_hold_baseline, run_equal_weight_baseline
from src.evaluation.metrics import compute_evaluation_summary

def main():
    print("=== Launching Tier 2 Small Test ===")
    orchestrator = DailyOrchestrator(model_id="openai/gpt-4o-mini")
    
    # 1. Warmup window (3 months: 2018-01-02 to 2018-03-31)
    print("Running Warmup Window & Risk-Aversion Lambda Grid Search...")
    best_lambda = run_lambda_grid_search(orchestrator, start_date="2018-01-02", end_date="2018-03-31")
    print(f"Optimal Lambda Selected on Warmup: {best_lambda}")
    
    # Freeze lambda for test window
    orchestrator.risk_aversion_lambda = best_lambda
    
    # 2. Test window (1 month: 2022-01-02 to 2022-01-31)
    print("\nRunning Out-of-Sample Test Window...")
    df_test = orchestrator.run_daily_loop(start_date="2022-01-02", end_date="2022-01-31", is_warmup=False)
    
    # 3. Baselines evaluation
    price_pivot = get_pivot_close_prices(orchestrator.assets, start_date="2022-01-02", end_date="2022-01-31")
    bh_rets = run_buy_and_hold_baseline(price_pivot)
    ew_rets = run_equal_weight_baseline(price_pivot)
    
    # 4. Reporting
    print("\n================ Tier 2 Evaluation Benchmark ================")
    summary_agent = compute_evaluation_summary(df_test["Portfolio_Return"], model_name="Memory_LLM_Agent")
    summary_bh = compute_evaluation_summary(bh_rets, model_name="Buy_and_Hold")
    summary_ew = compute_evaluation_summary(ew_rets, model_name="Equal_Weight")
    
    import pandas as pd
    df_summary = pd.DataFrame([summary_agent, summary_bh, summary_ew])
    print(df_summary.to_string(index=False))
    print("\nTier 2 Small Test completed successfully!")

if __name__ == "__main__":
    main()
