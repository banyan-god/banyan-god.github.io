---
title: "Many Contexts Is All You Need"
date: 2026-04-30
description: "RLM — Recursive Language Models. The model calls itself with smaller problems and fresh working memory. Here's why this changes everything."
author: "Sabareesh"
draft: false
tags: ["RLM", "AI agents", "LLM", "context window", "recursive delegation", "open source"]
categories: ["Artificial Intelligence"]
---

In December 2025, Alex Zhang and collaborators at MIT published a paper called [Recursive Language Models](https://github.com/alexzhang13/rlm). The core result: GPT-5 scores 0% on a retrieval task over 1000 documents (10M+ tokens). RLM wrapping the same GPT-5 scores 91.3%. Same model. The difference is how context is managed.

That result stuck with me. I've been building recursive agent systems on top of it since. This post is about what RLM actually is, why it works, and what I've learned running it on real tasks.

# The Problem: Context Rot

Context windows degrade before they fill up. The paper calls this "context rot" — as tokens accumulate, the model's performance on the task deteriorates. Not because it runs out of space, but because attention dilutes across noise: stale tool outputs, dead-end explorations, intermediate steps that are no longer relevant.

The industry's answer has been longer context. 128K, 1M, 10M tokens. This is like treating a cluttered desk by buying a bigger desk.

RLM says: don't put the data in the context window at all.

# The Key Insight: Symbolic Context

In a standard LLM call, context goes directly into the model's token window:

```
Standard:  LLM(query, context) → answer
```

The model sees the query and all the context as tokens. If the context is 10M tokens, the model needs a 10M token window and performance degrades across the whole thing.

RLM does something different. It loads the context into a variable in a code execution environment. The model never sees the raw data in its own window. Instead, it sees metadata — "you have a context variable with 10M characters" — and interacts with it through code:

```
RLM:  LLM(query, metadata) → code → REPL(code, context) → result
```

The model writes Python to slice, grep, chunk, and process the context programmatically. When it needs a sub-task answered, it calls `llm_query(prompt)` — which sends a subset of the data to a fresh LM call with a clean context window. The sub-call sees only what was passed to it. The root's context window only accumulates its own code and truncated outputs.

The context window IS the stack frame. The data lives outside, in the environment. Each `llm_query` call is a recursive function call with a fresh stack frame.

This is what gives RLM its power: **the root's context never grows with the data size.** A task over 10M tokens and a task over 100K tokens use roughly the same amount of root context, because the root is writing code to decompose the data, not ingesting it.

# Three Design Choices

The paper identifies three architectural choices that make RLM work:

**1. Symbolic context.** The data is a REPL variable, not tokens in the window. The model inspects it programmatically — `context[:5000]`, `re.findall(pattern, context)`, `context.split('\n')`. This means the model can handle inputs far larger than its context window, because it never loads the whole thing at once.

**2. Unbounded output.** Instead of generating the answer token by token, the model can build up results in REPL variables and return them via `FINAL_VAR(variable_name)`. This decouples output length from autoregressive generation limits.

**3. Symbolic recursion.** `llm_query()` can be called inside loops. The model can write:

```python
summaries = []
for chunk in chunks:
    summary = llm_query(f"Summarize: {chunk}")
    summaries.append(summary)
final = llm_query(f"Synthesize these summaries: {summaries}")
```

This is where the "recursive" in RLM earns its name. The model isn't just making one sub-call — it's programmatically constructing thousands of sub-queries in code, each receiving a fresh context window, each processing a precise subset of the data. The decomposition strategy emerges from the model's own code, not from a human-designed pipeline.

# What the Model Actually Does

The paper observes emergent strategies that models develop when given these tools:

- **Peeking** — inspect the first N characters to understand structure before committing to a strategy
- **Grepping** — use regex to filter relevant lines from a massive context
- **Partition + Map** — chunk the context, call `llm_query` on each chunk, combine results
- **Summarize + Decide** — compress subsets for high-level reasoning, then drill into specifics

No one programs these strategies. The model discovers them from the system prompt and the tools available. This is why the paper calls RLM "task-agnostic" — the same setup works for code analysis, document retrieval, distributional reasoning, and summarization.

# From REPL to Agents

The original RLM uses a Python REPL. Context is a Python variable. Sub-calls are `llm_query()` function calls. This is elegant for data processing tasks where the input is a long document or dataset.

But coding agents operate differently. Their "context" isn't a variable — it's the world. Files on disk, web pages, APIs, running processes. The agent's tools (bash, file I/O, web browsing) are the REPL. The same insight applies: **keep the root agent's context clean by delegating focused subtasks to fresh agents, each with their own context window.**

This is what I built with [pi-hydra](https://github.com/banyan-god/pi-hydra) — RLM implemented as an agent extension instead of a REPL wrapper. Instead of `llm_query(prompt)`, the agent calls `delegate(task)`. Instead of the context being a Python variable, it's accessed through tools. But the core mechanism is the same:

- Root's context stays clean — it only sees compressed results from children
- Each child gets a fresh context window focused on one subtask
- Children can recurse further (up to a depth limit)
- The decomposition strategy comes from the model, not from hardcoded logic

What pi-hydra adds beyond the paper's architecture:

**Async fan-out.** The original RLM supports batched `llm_query`, but pi-hydra goes further — truly async background processes with sentinel files for completion detection. Five agents researching in parallel, each on a separate context, each returning compressed results when done.

**Depth-dependent behavior.** The paper uses `max_depth=1` by default (root → sub-LM, no further recursion). Pi-hydra uses `max_depth=3` and encodes different behavioral modes at each depth. Depth 0 orchestrates. Depth 1 coordinates. Depth 2+ executes. The same model shifts from strategic planning to direct tool use based on one environment variable. Testing showed maxDepth=3 is optimal — at maxDepth=4, agents spent their budget re-delegating instead of doing actual work.

**Tree-wide resource tracking.** Every agent reads shared files that track cost, call count, and elapsed time across the entire delegation tree. An agent deep in the tree can see global state and decide whether to delegate or work directly based on remaining budget.

**Compression discipline.** The system prompt enforces a rule from the paper's core insight: each agent's return must be smaller than its input. Sub-agents compress — they extract the relevant facts and discard the noise. This is what keeps the root's context clean as results flow upward.

# Testing RLM: ASML Equity Research

To stress-test the system, I ran a real task: produce a buy/sell recommendation for ASML with deep multi-source research.

Setup: Qwen 3.6 27B served by vLLM on two consumer GPUs (NVIDIA 4090s). Root on GPU 1, children on GPU 2. $0 API cost.

The root followed the decomposition pattern naturally:
1. Delegated a scout to map the research landscape from seed URLs
2. Delegated a coordinator to fan out depth-2 workers across source categories
3. Workers browsed specific URLs, extracted data, compressed returns
4. Coordinator combined worker outputs into a compressed summary
5. Root synthesized everything into the final report

Output: 222-line equity research report. Earnings analysis (Q1 2026 beat: €8.8B revenue, 53% margins, guidance raised), moat assessment (100% EUV monopoly through 2030+), geopolitical risk (China revenue dropped from 36% to 19%), valuation (trailing P/E ~50x, forward ~35x), and a buy/sell verdict (BUY, moderate conviction, $1,500-$1,750 target).

Three things demonstrated RLM's properties:

**Compression worked.** Workers consumed 25-30K tokens of context each. Their returns were 200-600 tokens. The root's context stayed clean for synthesis — just the original task and a handful of distilled summaries.

**Depth-dependent behavior emerged.** The root never browsed a web page (in testing, when it did, it died from context exhaustion in 3/7 runs). The coordinator delegated but also did direct work. Workers only used tools. Same model, same weights, different cognitive mode at each depth.

**Graceful degradation.** 2 of 5 workers timed out (GPU throughput dropped from ~26 tok/s to ~5 tok/s with concurrent requests). The root detected timeouts, extracted partial results, and filled gaps itself. The report was complete despite 40% of workers failing.

A 27B model produced this. Not because 27B is enough for equity research — it isn't, in a single context. But the same 27B model, recursing across fresh context windows, with compression at every boundary? That works.

# Why This Matters

The original RLM paper showed something I think is underappreciated: **RLM(GPT-5-mini) outperformed base GPT-5 by 33%+ on distributional reasoning tasks.** A smaller model with recursive decomposition beat a bigger model processing everything at once.

This challenges the assumption that capability scales with model size and context length. It suggests a different axis of scaling: **the number of fresh context windows applied to a problem.**

Think about what this means for running agents locally. You don't need a 400B model with a 1M token window. You need a model that's sharp at 32K tokens — even a 27B model — and you give it 32K of clean, focused input every time. The capability comes from the recursive structure, not the parameter count.

The paper's cost analysis reinforces this: RLM(GPT-5) averaged $0.99 per query on a 10M-token retrieval task, versus $1.50-2.75 for GPT-5-mini processing equivalent tokens directly. More capable AND cheaper, because most of the work happens in small, focused sub-calls rather than one massive context window.

The binding constraint on agent performance isn't model intelligence or context size. It's context quality — how much of the working memory is signal versus noise. RLM attacks this directly: fresh windows, symbolic context, compression at every return boundary.

Same model. Same tokens. Different structure. Better results.

Many contexts is all you need.

---

**References:**
- [Recursive Language Models](https://github.com/alexzhang13/rlm) — Alex Zhang, Tim Kraska, Omar Khattab (MIT). The original paper and implementation.
- [pi-hydra](https://github.com/banyan-god/pi-hydra) — My implementation of RLM as a Pi coding agent extension. Async fan-out, depth-aware behavior, tree-wide tracking. MIT licensed.
- [ypi](https://github.com/rawwerks/ypi) — Another RLM implementation for Pi, shell-based, using jj workspaces for isolation.
