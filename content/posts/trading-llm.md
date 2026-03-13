---
title: "Teaching Qwen3-4B to Trade: From Hold-Collapse to +9.4% Returns"
date: 2026-03-13
description: "How I fine-tuned a 4B parameter LLM into a trading decision model using a 5-stage SFT waterfall and weighted CCE loss, then why I'm pivoting to GRPO reinforcement learning."
author: "Sabareesh"
draft: false
tags: ["LLM", "Fine-tuning", "GRPO", "Reinforcement Learning", "Trading", "Qwen3", "torchtune", "NeMo-RL", "vLLM"]
categories: ["Machine Learning", "Research"]
---

Can you turn an LLM into a profitable trader? I spent three months finding out. This post covers the full arc: a 5-stage supervised fine-tuning pipeline on Qwen3-4B, a catastrophic failure mode I had to diagnose and fix, the checkpoint that hit +9.4% returns with perfect format validity, and why supervised learning hit a ceiling that only RL can break through.

## The Setup

The model is Qwen3-4B. The task: given 30 days of OHLCV data plus 20+ quantitative features (RSI, MACD, volatility, beta, etc.) for a stock, output a structured JSON trade plan inside `<think>...</think><answer>{plan}</answer>` tags. The plan includes decision (enter/hold), side (long/short), stop loss, take profit, holding days, and position size.

Training runs on 2x NVIDIA RTX PRO 6000 Blackwell GPUs (96GB VRAM each). Evaluation uses vLLM for batched inference against a 1024-record portfolio simulation spanning 2024-2025.

The dataset: 5,176 equity reports, 103,343 stock prediction records across 94 symbols and 5 years of history, all exported from SQL Server through a custom DI-container pipeline.

## The 5-Stage Waterfall

I trained in stages, each building on the last:

1. **Equity knowledge injection** -- teach the model financial language from equity research reports
2. **Alpaca instruct alignment** -- make it follow instructions cleanly
3. **Stock prediction** -- next-day close prediction (~5-6% MAPE)
4. **Stateful trading decisions** -- the core task, outputting structured trade plans
5. **Distributed training** -- scaling across GPUs for larger runs

Each stage used torchtune with the previous stage's checkpoint as the base. The idea is progressive specialization: don't ask a base model to learn financial reasoning and structured output and trading decisions all at once.

## The Hold-Collapse Problem

By epoch 5, the model had learned the output format perfectly -- 512/512 valid JSON outputs. But it was entering trades on 99% of records. Clearly overfitting to the "enter" class after I'd fixed a format issue.

By epoch 18, the opposite happened: **100% valid format, 0% enter rate.** The model had collapsed to always outputting "hold." This is the trivial safe solution -- the training data is majority hold (the market isn't always signaling entry), so the model learned that always predicting hold minimizes loss.

This is the **hold-collapse failure mode**, and it's nasty because traditional metrics look fine. Format validity is perfect. Training loss converges. The model just learned the wrong thing.

## The Fix: Weighted Cut Cross-Entropy

The key insight: not all tokens in the output matter equally. The "decision" field matters enormously. The "think" block is just reasoning scaffolding. And within the decision-dependent fields, enter trades need much more emphasis than hold trades (which the model already does well).

I built a **conditional weighted CCE loss** using Apple's Cut Cross-Entropy library (4.7x faster than chunked CE, 3.2x less memory). The weights are per-field AND per-sample:

| Field | Enter Weight | Hold Weight |
|-------|-------------|-------------|
| decision | 8.0 | 8.0 |
| side | 6.0 | 1.5 |
| stop_loss | 4.0 | 0.75 |
| take_profit | 3.5 | 0.75 |
| max_holding_days | 4.0 | 1.5 |
| position_size | 3.0 | 0.75 |
| think block | 0.0 | 0.0 |

Sample-level multipliers: `enter_selected` at 2.5x, `enter` at 1.75x, `hold` at 1.0x.

I also discovered that **epoch 12 was a better recovery base than epoch 15 or 18**. Epoch 12 still had a 10.5% enter rate versus epoch 15's 0.5% -- it hadn't fully collapsed yet.

## The Gold Checkpoint

Training the v2 weighted CCE from epoch 12:

| Metric | Result |
|--------|--------|
| Final Equity | **$109,428** (+9.4%) |
| Enter Rate | 13.3% |
| Format Validity | 100.0% |
| Invalid Plans | 0 |
| Executed Trades | 136 |

This was the best result across 12 experimental runs. Perfect format, reasonable trade frequency, and actual portfolio growth.

## What I Tried That Didn't Work

| Experiment | Result | Why It Failed |
|-----------|--------|---------------|
| Equal class balancing (hold = buy = sell) | $100,386, 6.15% enter | Over-steered the policy, too few trades |
| 2:1 hold ratio | $96,266, 21.19% enter | Over-trading, 12 invalid plans |
| Clean dataset continuation (5y) | $96,273 | Distribution shift from legacy-calibrated weights |
| Clean dataset continuation (10y) | $104,205, 2.15% enter | Severe undertrading -- lost boundary supervision |
| Fresh train from raw epoch 12 | $89,649 | Perfect format but worst economic outcome |
| Low learning rate (1e-6) | $95,988, 1.95% enter | Too conservative, only 20 trades |

The most interesting failure: removing `enter_skipped_capacity` rows from the dataset. These are records where the oracle *wanted* to enter but couldn't because the portfolio was at capacity. They're labeled "hold" but carry an enter-like market context. Removing them stripped the model of critical **boundary supervision** -- the ability to distinguish "no edge, hold" from "edge exists but can't act." The enter rate collapsed from 13.3% to 2.15%.

## The Full Leaderboard

| Rank | Equity | Enter Rate | Validity | Run |
|------|--------|-----------|----------|-----|
| 1 | $109,428 | 13.28% | 100.0% | **v2 weighted CCE (gold)** |
| 2 | $104,205 | 2.15% | 100.0% | v2 + clean 10y |
| 3 | $102,220 | 13.28% | 99.9% | v1 weights |
| 4 | $100,951 | 11.52% | 99.4% | v2 + boundary supervision |
| 5 | $100,386 | 6.15% | 98.5% | equal class balancing |
| 6 | $100,182 | 0.49% | 100.0% | epoch 15 recovery |
| 7 | $96,273 | 13.96% | 99.4% | v2 + clean 5y |
| 8 | $96,266 | 21.19% | 98.8% | hold 2:1 ratio |
| 9 | $95,988 | 1.95% | 99.9% | low LR |
| 10 | $89,649 | 9.86% | 100.0% | raw epoch 12 fresh |

Oracle baseline: $197,585 at 13.7% enter rate.

## Why SFT Hit a Ceiling

The model imitates the oracle at ~80% decision match rate. But it captures less than 10% of the oracle's alpha. Why? **The oracle cheats.** It uses future price data to make decisions. The model only sees the 30-day lookback window.

SFT trains the model to mimic oracle labels, but those labels are computed from information the model will never have at inference time. The model can learn the *format* and the *style* of good trades, but it can't learn the actual edge because the edge comes from future information.

This is the fundamental limitation of imitation learning for trading.

## The RL Pivot

The solution: stop imitating the oracle and start optimizing the model's own strategy through reinforcement learning.

I'm implementing GRPO (Group Relative Policy Optimization) with a volatility-normalized reward function bounded to [-2, +2]:

- **For enters**: `0.8 * pnl_score - 0.2 * drawdown_penalty` where pnl_score uses `tanh(return / sigma_h)` to normalize by stock-specific volatility
- **For holds**: Counterfactual opportunity regret -- only penalizes missing moves larger than 1.25 sigma (obvious trades)
- **Invalid format**: -2.0 floor
- **Infeasible sizing**: -1.5

No enter bonuses, no hold penalties, no shaped rewards that could create degenerate optima. The reward comes purely from whether the trade was good relative to what the market actually did.

The infrastructure is already built: NeMo-RL with Ray + vLLM for rollouts, TRL as a parallel track. The v2 checkpoint is the starting point -- 100% format validity means RL can focus on improving *what* to trade rather than *how* to format the output.

I surveyed the literature before designing the reward. Trading-R1 (Sharpe 2.72 on NVDA) validates the exact SFT-to-RL progression I'm following. FLAG-Trader shows LLMs work as direct RL policy networks with LoRA + PPO. DAPO at the FinRL Contest 2025 achieved 230% cumulative return on NASDAQ-100.

## What's Next

The endgame is what I'm calling the **Living Trader** architecture -- four loops running at different frequencies:

1. **Inference loop** (real-time): model makes decisions from live data
2. **Experience collection** (daily): log predictions, outcomes, and counterfactuals
3. **RL training** (weekly): GRPO updates from collected experience
4. **Self-research** (continuous): the system identifies its own failure modes and proposes experiment modifications

For now, the immediate next step is Phase 1 GRPO calibration: 500 steps at temperature 0.4-0.5 to establish baseline reward distributions before the main 5,000-step run.

The v2 checkpoint proved that SFT can build a structurally sound trading model. RL is how we teach it to actually make money.

---

*Training hardware: 2x NVIDIA RTX PRO 6000 Blackwell (96GB each), 4x RTX 4090. Stack: torchtune, TRL, NeMo-RL, vLLM, llama.cpp. All experiments logged in W&B with formal journaling and phase gates.*
