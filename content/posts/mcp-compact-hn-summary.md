---
title: "MCP Compact: StreamableHTTP proxy that summarizes upstream tool output"
date: 2025-11-20
description: "What it does: intercepts MCP tool responses, summarizes them per-tool, and keeps sessions alive after upstream drops."
author: "Sabareesh"
draft: true
tags: ["summary"]
---

One-liner for HN: MCP Compact is a StreamableHTTP MCP proxy that summarizes tool outputs per-tool (using configurable token budgets/preservation instructions) and auto-reconnects when the upstream session drops. Keeps agents within context limits without changing agent code.
