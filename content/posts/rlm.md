---
title: "Many Contexts Is All You Need"
date: 2026-04-30
description: "RLM — Recursive Language Modeling. Same model, same tools, calling itself with smaller problems and fresh working memory."
author: "Sabareesh"
draft: false
tags: ["RLM", "AI agents", "LLM", "context window", "recursive delegation", "open source"]
categories: ["Artificial Intelligence"]
---

Context windows are working memory. Every tool output, every file read, every dead-end exploration — it accumulates and can't be forgotten. By turn 20, the signal that matters is buried. The model isn't dumber. It's distracted.

The industry answer is longer context. 128K, 1M, 10M tokens. This treats the symptom. The disease is that working memory fills with noise faster than you can use it.

RLM starts from a different place: **what if context is not a fixed budget but a renewable resource?**

# A Recursive Function, Not a Pipeline

In programming, a recursive function calls itself with a smaller input. Each call gets a fresh stack frame — its own local variables, its own clean state. The only thing that crosses the boundary is the return value.

RLM applies this to language model agents. The agent calls itself with a smaller task. The child gets a fresh context window — its own working memory, completely empty. The only thing that flows back to the parent is a compressed result.

This is not multi-agent orchestration. In a multi-agent system, you design different agents with different roles — a searcher, an analyzer, a writer. In RLM, there is one agent. The same model, the same tools, the same system prompt. It calls itself. The specialization isn't designed — it emerges from two things: the task each copy receives, and where it sits in the call tree.

```
f(problem)
├── f(subproblem A)   ← fresh stack frame
├── f(subproblem B)   ← fresh stack frame
│   ├── f(sub-sub B1) ← fresh stack frame
│   └── f(sub-sub B2) ← fresh stack frame
└── f(subproblem C)   ← fresh stack frame
```

A conventional agent is an iterative loop: act → observe → act → observe, accumulating everything in one context window that gets noisier with every cycle. An RLM agent is a recursive function: each call gets a clean slate, does focused work, and returns a compressed value. The parent never inherits the child's mess.

The context window IS the stack frame.

# Compression at Every Boundary

The property that makes recursion useful in programming is that you only pass return values between frames, not the entire state. RLM works the same way.

A child agent might visit 15 web pages, try 3 dead-end approaches, and generate 30K tokens of tool output in its context. The parent sees none of that. It gets back maybe 400 tokens: the distilled finding, the key numbers, the conclusion.

This compression boundary is where noise gets killed.

Each level of the call tree is a compression step. Raw data enters at the leaves — the actual web pages, file contents, tool outputs. Each level refines it: drops irrelevant details, keeps the signal, returns a tighter summary. By the time information reaches the root, it's been through multiple rounds of distillation by agents that each had full reasoning capability and task awareness when they decided what to keep and what to drop.

The root's context ends up containing the original task and a handful of clean summaries. It's practically empty. It does the final synthesis with maximum attention on minimum noise.

This only works if returns are smaller than inputs. If a child dumps its entire context back to the parent, you've just moved the noise up a level. The rule is explicit: **each agent's return must be smaller than its input.** Agents compress — they extract the relevant facts and discard the rest.

# Depth Changes How the Same Model Thinks

This is the part that surprised me most. The same model, given the same tools and the same system prompt, behaves fundamentally differently based on one variable: its depth in the recursion.

**Depth 0 — the strategist.** The root agent reads the task and thinks about structure: what are the independent parts? What depends on what? It decomposes, delegates, waits for returns, synthesizes. It almost never touches a tool directly. In testing, when the root browsed the web itself — reading raw HTML, consuming 30K tokens of page content — it died from context exhaustion in 3 out of 7 runs. The root's job is to think, not to act. Its context is too valuable to fill with raw data.

**Depth 1 — the coordinator.** This agent owns one domain. It might receive "analyze the competitive moat" and go deep on that single dimension. If the domain is still too broad, it decomposes further and delegates to depth 2. But it also does real work — using tools, reading pages, running computations. It's the bridge between planning and execution.

**Depth 2+ — the worker.** At this depth, the delegate tool isn't even registered. The agent can't recurse further. It picks up tools — bash, web browser, file I/O — and executes. Every leaf node is doing direct work with a clean context focused on a narrow task.

This gradient from planning to execution isn't encoded in different prompts or different agent definitions. It's the same agent, the same weights, shifting its behavior based on a single environment variable (`DELEGATE_DEPTH`). Shallow calls think. Deep calls act. The recursion naturally creates a division of cognitive labor.

An interesting finding from testing: maxDepth=3 is the sweet spot. At maxDepth=4, agents spend their budget re-delegating instead of doing actual work. Zero browsing happened in those runs. Adding depth doesn't add capability — it adds overhead. Three levels (orchestrate → coordinate → execute) is the right structure.

# The Evaluate-and-Recurse Loop

RLM is not "fan out and collect." The core pattern has five steps:

**Size up → Decompose → Delegate → Combine → Evaluate**

The fifth step is what makes it recursive rather than just parallel. After combining results, the agent evaluates: is this complete? What gaps remain? What follow-up questions did the results raise?

If the first round of research missed something — maybe the geopolitics agent timed out, or the financial data was stale — the root doesn't just ship an incomplete report. It decomposes the gaps, delegates again, and fills them. If a sub-agent's return raises a new question that wasn't in the original decomposition, the root can delegate that question to a new agent.

This is iterative deepening, not one-shot fan-out. The recursion continues until the root decides the result is sufficient — or until the budget, timeout, or call limit says stop.

# Tree-Wide Awareness

Every agent in the tree can see the global state: how much budget is left, how many delegation calls have been made, how much time remains. This isn't a detail — it changes how agents behave.

An agent at depth 1 might check `tree_status` and see that 18 of 20 allowed calls have been used. It won't delegate further — it'll do the remaining work itself. Or it sees that only 60 seconds remain on the timeout and decides to return what it has rather than starting a new research thread.

This awareness is implemented through shared files on disk — no coordination server, no message passing. Cost is tracked in a shared append-only JSONL file. Call count is tracked by appending lines to a shared file (one line = one call). Timeout is computed from a shared start time. Every agent reads the same files.

The result is that agents across the entire tree — which may be running on different GPUs, different processes, potentially different machines — make decisions based on the same global resource picture. An agent deep in the tree won't burn the last of the budget on a low-priority subtask because it can see what the tree has already spent.

# Why Not Just Long Context?

Three specific mechanisms work against long context that RLM sidesteps:

**Attention dilution.** Transformer attention distributes across all tokens. As context grows, each token gets proportionally less attention. Important information from early in the conversation competes with stale tool output from twenty turns ago. RLM keeps every context short — each agent has its task at the top of a fresh window, in the highest-attention position.

**Accumulation without forgetting.** A context window can't selectively discard. Every failed approach, every irrelevant intermediate step stays present. In RLM, the child's entire context — including all its noise — is discarded when it returns. Only the compressed result survives. The noise stays local to the stack frame that produced it.

**Serial degradation.** In a single long-context run, each task makes the next task harder because the context is noisier. Task 5 has all the garbage from tasks 1-4 in its working memory. In RLM, tasks run in separate contexts. Task 5's context is just as clean as task 1's. There's no degradation across the tree because there's no shared context to degrade.

Long context is the right tool when every step depends on the last — sequential chains of thought where the model genuinely needs to see everything at once. RLM is the right tool when the problem has independent parts that each benefit from focused, clean attention.

# What This Gets You in Practice

I ran RLM with Qwen 3.6 27B — a 27B parameter open model — on two consumer GPUs (NVIDIA 4090s). Root on one GPU, children on the other. $0 API cost.

Task: produce a buy/sell recommendation for ASML with deep multi-source research.

The root delegated two tasks: a scout (map the research landscape from seed URLs) and a deep-dive coordinator. The coordinator fanned out to depth-2 workers — one per source category — each with explicit URLs and extraction commands. Workers browsed, extracted, compressed. The coordinator combined their returns. The root synthesized everything into a 222-line equity research report: earnings analysis, moat assessment, geopolitical risk, valuation multiples, analyst sentiment, bull/bear cases, and a buy/sell call.

Three things happened that demonstrate RLM's properties:

**Compression worked.** Workers consumed ~25-30K tokens of context each (web pages, tool outputs). Their returns were 200-600 tokens each. The coordinator compressed further. The root's context stayed clean for synthesis.

**Depth-dependent behavior emerged.** The root never browsed. The coordinator delegated but also did direct work. The workers only used tools. Same model, different cognitive modes at each depth.

**Graceful degradation.** 2 of 5 workers timed out (the GPU couldn't sustain 5 concurrent inference streams fast enough). The root detected the timeouts, extracted partial results from what completed, and filled the gaps itself from the coordinator's summary. The report was complete despite 40% of workers failing.

A 27B model can't do this in a single context. The report would be shallow, the later sections would degrade as context filled, and a single timeout or error would kill the whole run. RLM made it possible — not by being smarter, but by being structured.

# Why This Structure Isn't Arbitrary

Organizations work like this. A CEO doesn't read every customer email — information flows upward through layers of compression. Team leads summarize for managers, managers for directors, directors for executives. Each layer drops noise and passes signal. Decisions happen at the top based on distilled input, not raw data.

This structure exists because human working memory is bounded (~7 items). The hierarchy is a recursive solution: each person handles what fits in their working memory, compresses the result, and passes it up. The tree isn't an org chart convention — it's the natural shape of bounded-memory information processing.

RLM is the same pattern applied to language models. The root is the executive. The leaves are individual contributors. The middle layers are managers. The compression boundaries are the reporting structure.

The interesting implication: the right number of levels isn't "as many as possible." In organizations, too many management layers slow things down — information gets over-compressed, decisions get disconnected from reality. Same with RLM: maxDepth=4 produced worse results than maxDepth=3 because agents spent their budget coordinating instead of working. Three levels (orchestrate → coordinate → execute) mirrors the structure of effective small teams.

# Tradeoffs

**Compression is lossy.** Each boundary drops details. If a worker misjudges what matters and cuts the wrong thing, the parent can't recover it. Quality depends on compression quality, not just reasoning quality.

**Decomposition bounds the result.** If the root splits the problem poorly — creates hidden dependencies between subtasks, misses a dimension, splits too fine or too coarse — the synthesis can't fix it. The quality of the recursive decomposition is the ceiling.

**Inference cost scales with the tree.** Every node needs compute. On one GPU, concurrent agents compete for throughput (~26 tok/s drops to ~5 tok/s with 5 concurrent requests). Multi-GPU is the fix but requires hardware. On APIs, every agent costs tokens.

**Sequential tasks don't decompose.** When step 2 depends on step 1, recursion adds overhead for no benefit. RLM is for problems with independent parts.

# The Point

The industry is scaling context length. RLM scales context count.

Long context: "I need to see more at once."
RLM: "I need to think clearly about each part."

Same model. Same tokens. Different structure. The capability comes from the recursion — fresh stack frames, compression at every boundary, depth-dependent cognition, and an evaluate-and-recurse loop that iterates until complete.

Many contexts is all you need.

---

One implementation: [pi-hydra](https://github.com/banyan-god/pi-hydra) — RLM as a Pi extension. Self-invocation, async fan-out, depth-aware behavior, tree-wide cost/call/timeout tracking, shared sessions between siblings, git worktree isolation. MIT licensed.
