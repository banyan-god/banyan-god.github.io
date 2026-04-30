---
title: "Many Contexts Is All You Need"
date: 2026-04-30
description: "RLM — Recursive Language Modeling. Why many fresh context windows beat one long one."
author: "Sabareesh"
draft: false
tags: ["RLM", "AI agents", "LLM", "context window", "recursive delegation", "open source"]
categories: ["Artificial Intelligence"]
---

Context windows are working memory. They degrade long before they fill up.

Every agent I've run on a complex task follows the same arc: sharp for the first few turns, then progressively worse. Not dumber — noisier. By turn 20, the context is 80% old tool outputs, dead-end explorations, and files that were relevant three steps ago. The model is spending its attention on junk.

The industry solution is bigger windows. 128K, 1M, 10M. This is like treating a cluttered desk by buying a bigger desk. You still can't find anything.

RLM takes the opposite approach: **more desks, not bigger ones.**

# Recursive Language Modeling

An RLM agent doesn't try to hold everything in one context. It breaks the task apart, hands each piece to a fresh copy of itself, and combines the compressed answers.

![RLM Architecture](/blog-images/rlm-architecture.svg)

Each sub-agent starts with an empty context. It gets one job description, nothing else. No history from the parent. No leftover outputs from sibling tasks. Its full attention budget goes to the actual work.

When it finishes, it doesn't return its entire context. It returns a summary. This is the critical part — the **compression boundary** between parent and child. Raw data enters at the leaves, gets distilled at each level, and arrives at the root as clean, synthesis-ready input.

The root agent's context contains the original task and a handful of compressed summaries. It's practically empty. It does the easiest job — combining clean inputs — with the freshest context in the entire tree.

# Why Compression Boundaries Matter

A sub-agent researching ASML's Q1 earnings might read 15 articles, follow 3 dead ends, and produce 6 tool outputs. The parent sees none of that. It sees: "Revenue €8.8B, beat estimates by 3.5%, margins 53%, guidance raised."

That's 20 tokens instead of 20,000. And those 20 tokens are pure signal.

This is what separates RLM from "just use a long context." In a long context, everything is equally present — the model has to figure out what to attend to in a sea of tokens. In RLM, the compression already happened. Each boundary is a filter that drops noise and keeps signal. The deeper the tree, the more refined the information that reaches the root.

# Depth Changes Behavior

The recursion has a structure:

**Depth 0** is the coordinator. It reads the task, decides how to break it up, delegates, waits, synthesizes. It never does raw work — reading files, searching the web, running tools. That would pollute its context. It stays clean for the final synthesis.

**Depth 1** is the specialist. It owns one piece — financials, or competitive analysis, or geopolitical risk. It goes deep on that one thing. If the piece is still too broad, it can delegate further.

**Depth 2+** is the worker. No more delegation. It picks up tools — bash, web search, file I/O — and executes. Every leaf node in the tree is doing direct work with a clean context focused on a narrow task.

The rule: the deeper you go, the more you work with your hands. Shallow agents think. Deep agents act. Without this, agents just keep spawning copies of themselves in an infinite loop of delegation.

The recursion is bounded — max depth (typically 3), a call budget, a timeout. At the limit, the delegate tool disappears. The agent has no choice but to work directly.

# What This Gets You

**Small models doing complex work.** A 27B model can't research a stock in a single pass — not enough context, not enough focus. But five copies of it, each researching one dimension with a fresh context? That works. I ran Qwen 3.6 27B on two consumer GPUs, five parallel delegates, and got a 222-line equity research report with earnings analysis, moat assessment, geopolitical risk, valuation multiples, and a buy/sell call. $0 API cost.

**Parallel execution.** Five agents on five tasks simultaneously, each with full context budget. A single agent doing the same five tasks sequentially gets worse at each one because its context is accumulating the output of all previous tasks.

**Failure isolation.** If one sub-agent times out or produces garbage, the parent notices and works around it — retries, fills the gap itself, or proceeds with partial data. In a single long-context run, one bad stretch of reasoning contaminates everything that follows. There's no rollback.

**Model-agnostic scaling.** You don't need a model trained on 1M tokens. You need a model that's sharp at 32K and you give it 32K of clean input every time. RLM works with APIs, open-weight models, local inference — anything that can follow instructions and use tools.

# The Tradeoffs

**Compression is lossy.** The parent trusts each child's judgment about what mattered. A bad summary propagates bad conclusions upward with no way to recover the dropped details.

**Decomposition is a skill.** The root has to break the task into genuinely independent pieces. Hidden dependencies between subtasks — where the valuation depends on the geopolitical analysis, say — create gaps the synthesis can't bridge.

**Not everything parallelizes.** Sequential reasoning where step 2 depends on step 1 doesn't decompose. RLM is for tasks with independent parts: research, multi-file refactors, analysis, review. For a single chain of thought, long context is exactly what you need.

**Inference cost.** Each agent needs its own compute. On one GPU, concurrent agents compete for throughput — I saw 26 tok/s drop to 5 tok/s with five concurrent requests. The fix is multi-GPU: root on one, children on another. On APIs, you're paying per token for every agent in the tree.

# The Point

We're solving the wrong problem. The bottleneck isn't how much the model can hold — it's how clean its working memory is. A fresh 32K context with focused input beats a polluted 1M context. Same model, same tokens, different signal-to-noise ratio.

RLM exploits this. Spawn fresh contexts. Compress across boundaries. Let every model in the tree work with a clean slate.

Many contexts is all you need.

---

One implementation: [pi-hydra](https://github.com/banyan-god/pi-hydra) — RLM as a Pi extension. MIT-licensed, async fan-out, depth-aware behavior, tree-wide cost tracking.
