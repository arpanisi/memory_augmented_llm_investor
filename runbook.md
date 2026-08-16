# Memory-Augmented LLM Investor Runbook

This project runs from `investor/memory_augmented_llm_investor`.

---

## 1. Overview & Data Contract

### Target Outcome
Build a memory-augmented LLM trading-agent framework targeting a benchmark across LLM backbones on a joint 15-asset portfolio (10 US equities, 2 crypto, 3 ETFs). Bounded by a 14-factor Barra-style risk model (10 industry Divisions + 4 style factors estimated via daily cross-sectional WLS regression on a 300-stock universe), point-in-time SEC EDGAR fundamentals, 4-layer persistent memory store (*short, mid, long, reflection*), and convex portfolio optimization (`cvxpy`, Ledoit-Wolf shrinkage, $\|w\|_1 \le 1$ budget).

### Data and Output Contract
- `data/` holds local cached data (`prices_cache.parquet`, `edgar_cache/`, `rf_daily.csv`, `barra_factor_returns.parquet`). Ignored by Git; only `.gitkeep` tracked.
- `outputs/` holds generated benchmark reports, evaluation tables, and performance CSVs. Ignored by Git; only `.gitkeep` tracked.
- `src/` holds all implementation code.
- `tests/` holds all automated unit tests.

---

## 2. Environment Setup

The project runs using the local Python virtual environment `.venv`.

### Virtual Environment Creation & Dependencies
```bash
# Create venv if not present
python3 -m venv .venv

# Activate environment and install dependencies
./.venv/bin/python -m pip install -r requirements.txt
```

### Environment Configuration (`.env`)
Create or edit `.env` inside `investor/memory_augmented_llm_investor/`:
```env
OPENROUTER_API_KEY=sk-or-v1-...
DEEPSEEK_API_KEY=sk-...
VLLM_BASE_URL=http://localhost:8000/v1
```

---

## 3. Automated Test Verification

Before running execution tiers, verify system components via pytest:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest tests/
```

Expected output: `58 passed`.

---

## 4. Execution Tiers

### Tier 1 — Smoke Test (Local Plumbing Check)
Exercises data materialization, memory layer insertion, view formation, portfolio optimization, breadth reporting, and reflection generation over 3 assets (`AAPL`, `BTC-USD`, `SPY`) for 10 trading days.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier1_smoke.py
```

### Tier 2 — Small Test (15-Asset Portfolio & Grid Search)
Full 15-asset universe, 3-month Warmup window (2018-01-02 to 2018-03-31), $\lambda \in \{0.5, 1, 2, 5, 10\}$ grid search on Warmup Sharpe, 1-month Out-of-Sample Test window (2022-01-02 to 2022-01-31), and baseline comparison against Buy-and-Hold and Equal-Weight.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier2_small.py
```

### Tier 3 — Full-Scale Backbone Benchmark
Runs full Warmup (2018–2021) and Test (2022–2023) windows across all 15 assets, sweeping view-formation backbones across OpenRouter (closed-source) and self-hosted vLLM (open-weight), outputting the final evaluation table to `outputs/backbone_benchmark_results.csv`.

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier3_full.py
```

---

## 5. Vast.ai GPU Deployment Guide (for Open-Weight vLLM Backbones)

For running open-weight view-formation backbones (e.g. `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`) and high-throughput embedding models on rented GPU instances via Vast.ai:

### Instance Provisioning on Vast.ai
1. Select an instance with an **NVIDIA A10G (24GB)**, **RTX 4090 (24GB)**, or **A100 (40GB/80GB)**.
2. Select an official PyTorch / CUDA 12.1+ Docker image (e.g. `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`).
3. Ensure port `8000` is exposed in the launch config.

### Host vLLM Inference Server on GPU Node
Connect via SSH to the Vast.ai instance and run:

```bash
# 1. Install vLLM
pip install vllm huggingface_hub

# 2. Set HuggingFace Access Token if downloading gated models
export HF_TOKEN="hf_..."

# 3. Launch vLLM OpenAI-Compatible API Server
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096
```

### Connect Investor Framework to Vast.ai Instance
On your local machine, update `.env` in `investor/memory_augmented_llm_investor/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
VLLM_BASE_URL=http://<VAST_AI_PUBLIC_IP>:8000/v1
```

Or run SSH port forwarding from local port `8000` to the Vast.ai GPU node:
```bash
ssh -L 8000:localhost:8000 -p <VAST_PORT> root@<VAST_HOST>
```

Then execute Tier 3 benchmark pointing to local forwarded endpoint:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_tier3_full.py
```
