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

How to run (local):
```bash
MCP_PROXY_CONFIG_FILE=config.json \
MCP_UPSTREAM_URL=http://localhost:8931/mcp \
BASE_URL=http://localhost:8080/v1 \
API_KEY=dev-key \
MODEL_NAME=openai/gpt-oss-120b \
python -m mcp_proxy --host localhost --port 8009
```

Tuning: edit `config.json` to pick which tools are summarized and how many tokens to keep. Leave a rule disabled to passthrough unchanged.

If you prefer containers:
```bash
docker compose up --build mcp-proxy
```

That is it—drop the proxy in front of your MCP server and keep the agent focused on the important bits.

Extensible: rules are per-tool, so you can add new MCP tools or tighten/loosen budgets without touching the agent. The proxy is open source at https://github.com/banyan-god/mcp-compact.
