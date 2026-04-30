---
title: "Many Contexts Is All You Need"
date: 2026-04-30
description: "RLM: why spawning many fresh context windows beats stretching one long one, and how pi-hydra makes it work with any model — including a 27B on consumer GPUs."
author: "Sabareesh"
draft: false
tags: ["RLM", "AI agents", "LLM", "context window", "recursive delegation", "vLLM", "Qwen", "open source"]
categories: ["Artificial Intelligence", "automation"]
---

# Abstract

We present Recursive Language Modeling (RLM) — a pattern where an AI agent solves complex tasks not by using a longer context window, but by spawning sub-agents with fresh ones. Each sub-agent receives only its task, operates with full context budget on a clean slate, and returns compressed results to its parent. We implement RLM as [pi-hydra](https://github.com/banyan-god/pi-hydra), an extension for the Pi coding agent. On an ASML equity research task with 5 parallel research streams, a 27B parameter model running on consumer GPUs ($0 API cost) produced a 222-line investment report with buy/sell recommendation — demonstrating that context management, not model scale, is the binding constraint on agent performance.

# 1. The Problem: Long Context Is a Leaky Bucket

The industry is racing to build longer context windows. 128K. 1M. 10M tokens. The assumption: if the agent could just hold more, it would perform better.

In practice, the opposite happens. As context fills — tool outputs, file contents, dead-end explorations — signal-to-noise ratio degrades. The agent doesn't run out of space. It runs out of focus. A 1M-token window doesn't fix this. It just lets the agent drown slower.

We observe this consistently across tasks: multi-file refactors, deep research, codebase investigations. The agent starts sharp, then degrades. Not from lack of capability — from context pollution.

The core insight: **the unit of scaling is not the context window. It is the number of context windows.**

# 2. RLM: Recursive Language Modeling

RLM treats context windows as a renewable resource. Instead of stretching one, you multiply them.

```
Root (depth 0) — orchestrator, clean context
├── Head α (depth 1) — subtask A, fresh context
├── Head β (depth 1) — subtask B, fresh context
│   ├── Head β₁ (depth 2) — sub-subtask, fresh context
│   └── Head β₂ (depth 2) — sub-subtask, fresh context
└── Head γ (depth 1) — subtask C, fresh context
```

**Definition.** An RLM agent decomposes a task into independent subtasks, delegates each to a child agent with an empty context window, and synthesizes the compressed results. The process is recursive: children may delegate further, bounded by depth limits and resource budgets.

Each child receives only its task description. No conversation history. No accumulated tool outputs. 100% of its context budget is available for the actual work. The parent never sees raw material — only distilled output.

This is the key property: **information flows up compressed, not down accumulated.**

# 3. Why Many Contexts > One Long Context

**3.1 Signal-to-noise ratio.** A sub-agent researching ASML's Q1 earnings has its entire context devoted to that task. No leftover file contents from earlier steps, no stale tool outputs, no competing instructions. Fresh context = focused attention.

**3.2 Parallelism.** Five research streams running concurrently on five fresh context windows finish faster than one agent doing them sequentially — and each stream gets full context budget instead of 1/5th of a shared window.

**3.3 Graceful degradation.** If one sub-agent fails (timeout, bad output, model error), the parent detects it and compensates. A single long-context agent that corrupts its own state has no recovery path.

**3.4 Model-agnostic scaling.** RLM works with any model. You don't need a model trained on 1M tokens. You need a model that's good at 32K tokens, and you give it 32K of pure signal.

# 4. pi-hydra: Implementation

[pi-hydra](https://github.com/banyan-god/pi-hydra) implements RLM as a Pi extension in ~500 lines of TypeScript. Two tools:

**`delegate(task, async?, isolate?, provider?)`** — spawn a sub-agent with a fresh context window. Optionally async (background execution with sentinel file), isolated (own git worktree), or routed to a specific provider/GPU.

**`tree_status()`** — inspect depth, cost, call count, and remaining timeout across the entire delegation tree.

### 4.1 Async Fan-Out

The critical capability. Spawn N children concurrently:

```
delegate(task: "Research financials", async: true)  → sentinel_A
delegate(task: "Research moat",       async: true)  → sentinel_B
delegate(task: "Research geopolitics", async: true) → sentinel_C
```

All run in parallel. Root polls for sentinel files, reads compressed results, synthesizes. N fresh contexts working simultaneously.

### 4.2 Tree-Wide Guardrails

Recursive agents without limits are dangerous. pi-hydra enforces:

| Guardrail | Default | Mechanism |
|-----------|---------|-----------|
| Max depth | 3 | `delegate` tool hidden at limit |
| Max calls | 20 | Shared counter file |
| Budget | $5.00 | Shared cost ledger |
| Timeout | 600s | Wall-clock from tree root |

No coordination server. All state is shared via lock-free append-only files.

### 4.3 Depth-Aware Behavior

The system prompt encodes different strategies per depth:

- **Depth 0 (root):** Orchestrate. Decompose. Never fill context with raw data. Preserve budget for synthesis.
- **Depth 1 (coordinator):** Focused execution. Delegate only if 3+ independent subparts remain.
- **Depth 2+ (worker):** Leaf node. Direct action — bash, file I/O, web tools. No further delegation.

Deeper agents do more work. Shallower agents do more thinking. The recursion is productive at every level.

### 4.4 Shared Sessions

Every agent logs its session to a shared directory. Siblings can read what others found:

```bash
delegate-sessions list          # all sessions in the tree
delegate-sessions read --last   # most recent
delegate-sessions grep "ASML"   # search across all
```

Later agents build on earlier findings. No repeated work.

# 5. Experiment: $0 ASML Equity Research

### 5.1 Setup

Model: Qwen 3.6 27B, served via vLLM on two consumer GPUs (NVIDIA 4090). Root on GPU 1 (port 8001), children on GPU 2 (port 8002) via `DELEGATE_CHILD_PROVIDER=vllm2`. Total API cost: **$0.00**.

One critical configuration discovery: `thinkingFormat` must be `"qwen-chat-template"` when serving Qwen through vLLM. The alternative (`"qwen"`) sends the thinking parameter in the wrong format. vLLM accepts the request but the model never produces tool calls — the agent loop silently breaks. This single config value took hours to diagnose.

### 5.2 Task

> Produce a buy/sell recommendation for ASML stock with deep multi-source research.

### 5.3 Decomposition

The root spawned 5 async delegates:

| Head | Domain | Context |
|------|--------|---------|
| α | Financials | Q1 2026 earnings, margins, guidance vs. consensus |
| β | Competitive moat | EUV/High-NA monopoly, tech roadmap, threats |
| γ | Geopolitics | China export curbs, chip war, EU/FX impact |
| δ | Valuation | P/E, P/S, EV/EBITDA vs. history and peers |
| ε | Sentiment | Analyst calls, institutional moves, catalysts |

Five fresh context windows, each 100% focused on its research domain. Zero context shared between them.

### 5.4 Results

3/5 delegates completed within timeout. Root detected 2 timeouts (inference throughput bottleneck at ~5 tok/s under concurrent load), extracted partial outputs, and filled gaps itself. Graceful degradation — exactly as designed.

Output: 222-line equity research report.

- **Verdict: BUY — Moderate conviction**
- **12-month target: $1,500-$1,750** (5-22% upside)
- Covered Q1 2026 beat (€8.8B vs €8.5B est), High-NA at $370M/unit, installed base recurring revenue, China revenue decline, AI capex supercycle

A 27B model with many fresh contexts produced analyst-grade research. The model wasn't the bottleneck. Context was.

### 5.5 Throughput

| Config | Tokens/sec | Notes |
|--------|-----------|-------|
| 1 request, 1 GPU | ~26 | Baseline |
| 5 concurrent, 1 GPU | ~5 per request | Contention |
| Root on GPU1, children on GPU2 | ~26 each | No contention |

Multi-GPU fan-out is essential for practical RLM on local hardware.

# 6. Discussion

**Context is the bottleneck, not intelligence.** A 27B model with fresh context outperforms a frontier model drowning in 100K tokens of accumulated noise. RLM is a context management strategy, not a capability multiplier.

**The recursive pattern is natural.** Humans don't solve complex problems by holding everything in working memory. We decompose, delegate to specialists, and synthesize. RLM gives agents the same workflow.

**Local models are viable orchestrators.** Qwen 3.6 27B reliably decomposes tasks, makes tool calls, and synthesizes sub-agent output at $0 per run. For long-running research tasks that would burn $20+ in API credits, the economics are decisive.

**The unit of scaling is the context window, not the token count.** Instead of asking "how do we fit more tokens?", ask "how do we use more windows?" This reframes the entire architecture of agent systems.

# 7. Try It

[pi-hydra](https://github.com/banyan-god/pi-hydra) is MIT-licensed. Works with any model Pi supports — OpenAI, Anthropic, Google, or local models via vLLM/Ollama.

```bash
npm install pi-hydra
pi-hydra --provider vllm --model Qwen/Qwen3.6-27B "Research and summarize X"
```

Source, system prompt, and CLI tools: [github.com/banyan-god/pi-hydra](https://github.com/banyan-god/pi-hydra)
