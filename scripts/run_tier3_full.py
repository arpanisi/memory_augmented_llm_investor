"""
Tier 3 — Full-Scale Benchmark Script.
Full Warmup (2018-2021) and Test (2022-2023) windows across all 15 assets, comparing LLM backbones vs baselines.
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
    print("=== Launching Tier 3 Full Benchmark ===")
    
    # Target backbones across OpenRouter and vLLM
    target_backbones = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-haiku",
        "deepseek/deepseek-chat"
    ]
    
    benchmark_results = []
    
    for model_id in target_backbones:
        print(f"\n>>> Running Full Pipeline for Backbone: {model_id} <<<")
        orchestrator = DailyOrchestrator(model_id=model_id)
        
        # 1. Warmup & Grid Search
        best_lambda = run_lambda_grid_search(orchestrator, start_date="2018-01-02", end_date="2021-12-31")
        print(f"Optimal Lambda for {model_id}: {best_lambda}")
        orchestrator.risk_aversion_lambda = best_lambda
        
        # 2. Out-of-Sample Test Window
        df_test = orchestrator.run_daily_loop(start_date="2022-01-02", end_date="2023-12-31", is_warmup=False)
        summary = compute_evaluation_summary(df_test["Portfolio_Return"], model_name=f"Agent_{model_id}")
        benchmark_results.append(summary)

    # 3. Baselines Evaluation over Test Window
    print("\nRunning Baselines over Test Window...")
    price_pivot_test = get_pivot_close_prices(orchestrator.assets, start_date="2022-01-02", end_date="2023-12-31")
    bh_rets = run_buy_and_hold_baseline(price_pivot_test)
    ew_rets = run_equal_weight_baseline(price_pivot_test)
    
    benchmark_results.append(compute_evaluation_summary(bh_rets, model_name="Buy_and_Hold"))
    benchmark_results.append(compute_evaluation_summary(ew_rets, model_name="Equal_Weight"))

    import pandas as pd
    df_final = pd.DataFrame(benchmark_results).sort_values("annualized_sharpe", ascending=False)
    
    output_path = BASE_DIR / "outputs" / "backbone_benchmark_results.csv"
    df_final.to_csv(output_path, index=False)
    
    print("\n================ FINAL BACKBONE BENCHMARK REPORT ================")
    print(df_final.to_string(index=False))
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
