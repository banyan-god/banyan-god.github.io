---
title: "MCP Compact: Keep Agent Context Lean"
date: 2025-11-20
description: "A proxy that trims noisy tool outputs so MCP agents stay on track."
author: "Sabareesh"
draft: false
---

The problem: MCP agents return bulky tool outputs (screenshots, DOM dumps, network traces) and quickly blow past context limits. Downstream steps stall or get fuzzy because the signal is buried.

What MCP Compact does: it sits between your agent and the upstream MCP server, forwards every tool call, and summarizes the response with an LLM. You set per-tool rules (token budget, what to preserve), and the proxy enforces them automatically.

Why it is useful:
- Keeps context budgets sane without touching the agent code.
- Works over StreamableHTTP, so progress and streaming stay intact.
- Reconnects to the upstream if sessions drop, so long runs do not die mid-task.

Example: a `browser_navigate` call can return a 100k+ character DOM dump with every script tag and hidden div. MCP Compact keeps the key UI elements and text in a few thousand tokens so the agent stays focused instead of drowning in markup.
Extensible:
- Per-tool rules let you tighten/loosen budgets as you add new MCP tools.
- Swappable summarizer: any LLM endpoint works; only needs tokens and a model name.
- Structured logging instead of guesswork: you see token counts, clipping, and summary lengths in one place.
- Session resilience: automatic reconnect to the upstream keeps long chains alive.

How it sits in the flow:
```
Agent → MCP Compact → Upstream MCP Server
          ↓
      LLM Summarizer
```

More details, configs, and run commands live in the repo: https://github.com/banyan-god/mcp-compact.
