---
title: "Many Contexts Is All You Need"
date: 2026-04-30
description: "RLM: why giving AI agents many fresh context windows works better than giving them one really long one."
author: "Sabareesh"
draft: false
tags: ["RLM", "AI agents", "LLM", "context window", "recursive delegation", "open source"]
categories: ["Artificial Intelligence"]
---

# The Bet Everyone Is Making

The entire industry is betting on longer context windows. 128K, 1M, 10M tokens. The logic sounds right: if the agent could just remember more, it would do more.

I've been running agents on real tasks — researching stocks, refactoring codebases, investigating bugs across dozens of files — and I keep seeing the same thing. The agent starts great. Crisp reasoning, focused tool use. Then it gets worse. Not because it's dumb, but because its context is full of junk. Old tool outputs. Files it read 20 turns ago. Failed attempts it should have forgotten. The useful signal is buried.

A longer context window doesn't fix this. It just gives the agent more room to accumulate noise.

# The Core Idea

What if, instead of making the window longer, you just opened more windows?

That's RLM. Recursive Language Modeling. The agent doesn't try to hold everything in one context. It breaks the problem into pieces, hands each piece to a fresh copy of itself, and combines the answers.

Each copy starts clean. No history. No leftover junk. 100% of its attention goes to the one thing it was asked to do.

Think about how you'd research a stock. You wouldn't sit in one chair and try to remember everything simultaneously — earnings, competitors, geopolitics, valuation, analyst sentiment. You'd split it up. Maybe you read the earnings yourself, ask a friend who follows trade policy about the China risk, ask another friend who tracks valuations to run the numbers. Each person focuses on their piece. You combine it all at the end.

RLM does exactly this. The root agent is you. The sub-agents are your specialists. Each one gets a fresh brain focused on one thing.

# Why This Works

**Fresh context means focused attention.** When a sub-agent is researching ASML's Q1 earnings, its entire context is about that. No leftover file dumps from a previous step. No competing instructions. Just the task and the tools. This is why a small model with clean context often outperforms a big model with polluted context.

**Parallel beats sequential.** Five sub-agents running at the same time, each with a full context budget, finish faster and produce better results than one agent doing all five tasks back-to-back in a single window that's getting progressively noisier.

**Failure doesn't cascade.** If one sub-agent times out or produces garbage, the parent notices and works around it. Maybe it retries, maybe it fills the gap itself. In a single long-context agent, one bad stretch of reasoning corrupts everything that comes after. There's no recovery.

**Any model works.** You don't need a model trained on 1M tokens. You need a model that's sharp at 32K tokens — and you hand it 32K of pure signal every time. This means RLM works with open-weight models, local models, even a 27B parameter model on a consumer GPU. The model isn't the bottleneck. The context is.

# The Pattern

RLM follows a simple loop: **size up, decompose, delegate, combine, evaluate.**

1. **Size up** the task. Is it small enough to do directly? If yes, just do it. No recursion needed.
2. **Decompose** into independent pieces. Each piece should be self-contained — the sub-agent gets only the task description, nothing else.
3. **Delegate** each piece to a fresh agent. They run in parallel when possible.
4. **Combine** the results. Deduplicate, resolve conflicts, synthesize.
5. **Evaluate** — is it complete? If gaps remain, decompose the gaps and delegate again.

The recursion is bounded. You set a max depth (typically 3), a budget, a timeout. At the deepest level, agents stop delegating and do the work directly. This prevents infinite spawning and keeps costs predictable.

Depth changes behavior:
- **Depth 0** — the orchestrator. Breaks the problem apart, stays clean, synthesizes at the end.
- **Depth 1** — the specialist. Focused on one domain. Delegates only if its piece is still too big.
- **Depth 2+** — the worker. Does the actual research, writes the actual code, reads the actual files. No more delegation.

The deeper you go, the more direct work happens. The shallower you are, the more you think and coordinate.

# What Changes

RLM reframes how you think about agent architecture.

**The unit of scaling isn't the token count — it's the number of context windows.** Instead of asking "how do we fit more into one context?", ask "how do we use more contexts?" A 128K window with 10 fresh agents is more capable than a 1M window with one.

**Context management is the real problem.** We've been focused on making models smarter and context windows longer. But the binding constraint is often neither — it's how much noise accumulates in the working memory. RLM sidesteps the problem entirely by throwing away the noise and starting fresh.

**Small models become viable for complex tasks.** A 27B model can't do deep equity research in a single pass. But a 27B model that spawns five copies of itself, each researching one dimension, and then synthesizes the results? That works. I tested it — Qwen 3.6 27B on two consumer GPUs produced a 222-line ASML equity research report with a buy/sell recommendation, covering earnings, moat analysis, geopolitics, valuation, and sentiment. Total API cost: $0.

**Humans already work this way.** We don't solve complex problems by holding everything in working memory. We decompose, delegate, specialize, reconvene. Every organization is an RLM system — a tree of agents with bounded context, communicating compressed results upward. RLM just gives AI agents the same structure.

# The Tradeoffs

RLM isn't free. There are real costs.

**Latency.** Spawning sub-agents takes time. If you're running local models, each agent needs inference capacity. Five concurrent agents on one GPU will be slow. You either need multiple GPUs or you accept the throughput hit.

**Lossy compression.** When a sub-agent compresses its findings into a summary, details get lost. The parent never sees the raw data — that's the whole point — but it means the synthesis depends on the sub-agent's judgment about what matters. A bad summary propagates bad conclusions.

**Coordination overhead.** The root has to decompose well. If the subtasks aren't truly independent, you get gaps or overlaps. Decomposition is a skill, and some models are better at it than others.

**Not everything decomposes.** Sequential reasoning — where step 2 depends on step 1's output — doesn't parallelize. RLM shines on tasks with independent parts: research, refactoring, analysis, review. It doesn't help with a single chain of thought that must be followed step by step.

# Try It

RLM is a pattern, not a product. You can implement it in any agent framework that lets you spawn sub-processes.

One implementation: [pi-hydra](https://github.com/banyan-god/pi-hydra) — an RLM extension for the Pi coding agent. MIT-licensed, works with any model (API or local via vLLM/Ollama), includes async fan-out, tree-wide cost tracking, depth-aware behavior, and shared sessions between sibling agents.

The core thesis stands independent of any implementation: **many contexts is all you need.**
