---
title: "Many Contexts Is All You Need"
date: 2026-04-30
description: "RLM — Recursive Language Modeling. Treat context windows like stack frames: spawn fresh ones, recurse, compress results upward."
author: "Sabareesh"
draft: false
tags: ["RLM", "AI agents", "LLM", "context window", "recursive delegation", "open source"]
categories: ["Artificial Intelligence"]
---

Context windows are working memory. They don't just fill up — they degrade. By turn 20, an agent's context is mostly stale tool outputs and dead-end explorations. The signal that matters is buried under noise the model can't forget. Bigger windows don't fix this. They just let the decay happen slower.

RLM — Recursive Language Modeling — starts from a different premise: **context is not a fixed resource. It's a renewable one.** An agent can spawn a copy of itself with a completely empty context window, hand it a focused task, and get back a compressed result. That copy can do the same thing. The recursion continues until the task is small enough to solve directly.

This is not multi-agent orchestration. It's recursion. The same distinction that exists in programming between a `for` loop and a recursive function.

# Recursion, Not Orchestration

Multi-agent systems assign different roles to different agents. One agent searches, another analyzes, another writes. The roles are designed upfront. The agents are different.

RLM uses one agent. The same model, the same tools, the same system prompt. It calls itself with a smaller problem. The specialization isn't designed — it emerges from two things: the task description each copy receives, and its depth in the tree.

In computer science, a recursive function and an iterative loop can solve the same problem. But they handle state differently. An iterative loop accumulates state in variables that persist across iterations — and that state can get stale, corrupt, or bloated. A recursive function gets a fresh stack frame on every call. Each frame is clean. The only information that crosses frames is the return value.

RLM applies the same idea to agents. A conventional agent is an iterative loop: read, act, read, act — accumulating everything in one context window that gets progressively noisier. An RLM agent is a recursive function: each call gets a fresh context (stack frame), does focused work, and returns a compressed result (return value). The parent never inherits the child's mess.

The context window IS the stack frame.

# Compression Is the Mechanism

What makes recursion work in RLM isn't the spawning — it's what happens at the boundary between parent and child.

A child agent might read 15 articles, try 3 approaches, generate 30K tokens of tool output. The parent sees none of that. It sees a 400-token summary: the distilled finding, the key numbers, the conclusion. This compression boundary is where noise gets killed.

Each level of the tree is a compression step. Raw data enters at the leaves. Each level refines it — drops irrelevant details, keeps the signal, passes a tighter summary upward. By the time information reaches the root, it's been through multiple rounds of distillation. The root's context contains the original task and a handful of clean summaries. It's practically empty. It does the final synthesis with maximum attention on minimum noise.

This is the opposite of long context. In a long context, the model has to compress on the fly, deciding what to attend to in a sea of 100K tokens. It's implicit, approximate, and gets worse as context grows. In RLM, compression is explicit and happens at defined boundaries. Each boundary is a filter that a full agent — with reasoning, judgment, and task awareness — applied deliberately.

The return value is the compressed result. The stack frame is disposed of. The noise stays local.

# Depth Changes How the Model Thinks

Here's the part that surprised me. The same model, given the same tools, behaves fundamentally differently depending on its depth in the tree.

**At depth 0**, the agent is a strategist. It reads the task, thinks about structure: what are the independent parts? What depends on what? How should the work be divided? It decomposes, delegates, waits, synthesizes. It almost never touches a tool directly — doing so would pollute its context with raw data it doesn't need.

**At depth 1**, the agent is a specialist. It owns one domain and goes deep. It uses tools — web search, file reading, computation — but with tight focus. If its domain is still too broad, it decomposes further and recurses.

**At depth 2+**, the agent is a worker. It picks up tools and executes. No decomposition, no delegation. Just direct work on a narrow problem with a clean context.

This isn't three different agents with three different prompts. It's the same agent, adapting its behavior to its position in the recursion. Shallow agents think. Deep agents act. This gradient from planning to execution emerges naturally from bounded recursion with depth-aware guidance.

The same model, the same weights, exhibiting different cognitive modes based on where it sits in a recursive call stack. The depth parameter is doing surprisingly heavy lifting.

# Why Not Just Use a Longer Context?

Three specific mechanisms work against long context:

**Attention dilution.** Transformer attention distributes across all tokens. As context grows, each token gets proportionally less attention. Important information from turn 3 competes with tool output from turn 18. This is measurable — retrieval accuracy drops in the middle of long contexts even in models specifically trained for it.

**Positional advantage.** Early context gets disproportionate attention from positional encoding. Every RLM sub-agent exploits this — its task is always at the top of a fresh context, in the highest-attention position. A single long-context agent buries later tasks under layers of prior output.

**Accumulation without forgetting.** A context window can't selectively forget. Every failed approach, every irrelevant tool output, every intermediate step stays present and competes for attention. RLM handles this structurally: the child's context (with all its noise) is discarded. Only the return value survives.

Long context is the right tool for sequential reasoning where every step depends on the last. RLM is the right tool for problems with recursive structure — tasks that decompose into independent parts that each benefit from focused attention.

# The Organizational Parallel

This pattern isn't new. It's how every effective organization works.

A CEO doesn't read every customer email. Information flows upward through layers of compression — team leads summarize for managers, managers summarize for directors, directors summarize for executives. Each layer filters noise and passes signal. The CEO makes decisions based on distilled input, not raw data.

This structure exists because human working memory is bounded. Seven items, plus or minus two. Organizations are a recursive solution to the same constraint RLM addresses: individual working memory is finite, but the problems are bigger than any one working memory can hold.

RLM gives agents the same structure. The root is the executive — clean context, high-level synthesis. The leaves are individual contributors — deep domain work with focused attention. The middle layers are managers — coordinating and compressing.

The tree isn't an arbitrary architecture choice. It's the natural shape of bounded-memory information processing.

# What This Gets You

**Small models doing hard tasks.** A 27B model can't research a stock in one pass. But a 27B model that recurses — root decomposes into five specialists, one specialist decomposes further into two workers — produces a 222-line equity research report covering earnings, moat, geopolitics, valuation, and sentiment. Same model at every node. The capability comes from the structure, not the parameter count.

**Context quality over context quantity.** Every agent in the tree works with clean context tuned to its specific task. No agent is polluted by sibling work. The root synthesizes from compressed summaries, not raw data. Signal-to-noise ratio stays high at every level.

**Structural fault tolerance.** A failed child is a failed branch, not a failed run. The parent detects it and compensates — retries, fills gaps, proceeds with partial data. In a single-context agent, failure cascades forward through the entire remaining context.

# The Tradeoffs

**Compression is lossy.** Each boundary drops details. The parent trusts the child's judgment about what matters. A bad summary propagates bad conclusions upward, and the dropped details can't be recovered.

**Decomposition quality matters.** If the root splits the task poorly — creates dependencies between subtasks, misses an important dimension, splits too fine or too coarse — the synthesis can't recover. The quality of the recursive decomposition bounds the quality of the final result.

**Inference cost scales with the tree.** Each node needs compute. On local hardware, concurrent agents compete for GPU throughput. The fix is splitting across GPUs, but that requires hardware. On APIs, every agent in the tree costs tokens.

**Sequential tasks don't benefit.** When step 2 fundamentally depends on step 1's output, there's nothing to parallelize and the overhead of spawning and compressing is pure waste.

# The Thesis

The industry is scaling context length. RLM scales context count. These solve different problems.

Long context addresses: "I need to see more at once."
RLM addresses: "I need to think clearly about each part."

The binding constraint on agent performance today isn't model intelligence or context size. It's context quality — how much of the working memory is signal versus noise. RLM attacks this directly: every agent in the tree gets a clean slate, and the compression boundaries ensure the noise stays local.

Same model. Same tokens. Different structure. Better results.

Many contexts is all you need.

---

One implementation: [pi-hydra](https://github.com/banyan-god/pi-hydra) — RLM as a Pi coding agent extension. Async fan-out, depth-aware behavior, tree-wide cost/call tracking, shared sessions between siblings. MIT licensed.
