# Sabareesh Subramani

[https://sabareesh.com](https://sabareesh.com) | [LinkedIn](https://www.linkedin.com/in/sabaree/) | [hello@sabareesh.com](mailto:hello@sabareesh.com)

## Summary

ML researcher and CTO exploring how language models can learn, adapt, and make decisions in real-time. Current research spans three areas: training LLMs that keep learning during inference (TTT-E2E with second-order meta-gradients, 4.6–6.1% perplexity gains on Qwen3-4B), teaching LLMs to trade autonomously via reinforcement learning (GRPO on proprietary financial data, +9.4% portfolio return), and building multi-agent systems that reason and act in the real world. Pre-trained LLMs from scratch, contributed to Karpathy's open-source training codebases, and run continuous experiments on personal GPU infrastructure. 14 years shipping production systems as CTO — now applying that engineering depth to AI research, and founder of [Spicedust](https://spicedust.ai), a research lab building living models (reasoning, agents, continual learning, memory).

## Technical Skills

- **Research Areas:** Test-time training (TTT-E2E), meta-learning (FOMAML, second-order meta-gradients), reinforcement learning for LLMs (GRPO, DAPO, PPO), reward shaping for financial markets, multi-agent architectures, continual learning during inference
- **LLM Training:** PyTorch, torchtune, torchao (float8/4-bit quantized training), TRL, NeMo-RL (GRPO/DAPO), DDP, FSDP/FSDP2, torchrun, vLLM, llama.cpp, Cut Cross-Entropy
- **Model Architectures:** Qwen3-4B, Llama 2/3, DeepSeek R1, GPT-2/NanoGPT, custom Transformer encoders
- **RL for LLMs:** GRPO, DAPO, PPO, TorchRL, volatility-normalized reward functions, counterfactual opportunity regret, hold-penalty scheduling, reward shaping for trading
- **Agent Systems:** OpenAI Agents SDK, MCP (Model Context Protocol), multi-agent orchestration, context compaction, semantic memory, WandB Weave observability
- **ML Infrastructure:** WandB, HuggingFace Hub, CUDA, Ray, multi-GPU training (4x RTX 4090, 2x RTX PRO 6000 Blackwell), checkpoint management, vLLM evaluation pipelines
- **Languages:** Python, Java, Swift, C, C++, TypeScript, C#
- **Backend & Infrastructure:** FastAPI, Spring Boot/Cloud, SQLAlchemy, Docker, Kubernetes, Azure, AWS, SQL Server, Kafka, Elasticsearch

## AI & ML Projects

### Qwen3-4B Financial Trading Model — SFT + RL Pipeline

Can an LLM learn to make autonomous trading decisions — not by imitating an oracle, but by discovering its own strategy through reinforcement learning?

**Supervised Fine-Tuning (5-stage waterfall):**
- Designed a multi-stage knowledge injection pipeline: equity domain knowledge → instruct alignment → stock prediction → stateful trading decisions. Each stage builds on the previous, progressively teaching the model to reason about markets.
- Discovered and solved the hold-collapse problem — the model converged to a trivial all-hold policy due to class imbalance. Developed weighted CCE loss with conditional per-field weights (enter-critical fields weighted 4-8x) to recover meaningful trading behavior.
- **Key result:** 100% format validity, 136 trades, +9.4% portfolio return on 1024-record evaluation.

**Reinforcement Learning (current phase):**
- SFT hit its ceiling — the model imitates the oracle at 80% match rate but captures <10% of oracle alpha (the oracle cheats with future information). The question: can GRPO discover trading strategies that SFT cannot?
- Designed volatility-normalized reward functions with counterfactual opportunity regret — the model is penalized not just for bad trades, but for missing moves it should have taken.
- Studying Trading-R1, FLAG-Trader, HCAPO, and DAPO to understand how others are applying RL to financial LLMs.
- Designing a "Living Trader" architecture: real-time inference, daily experience collection, weekly RL training, continuous self-research.
- **Technologies:** torchtune, TRL, NeMo-RL (Ray + vLLM), Cut Cross-Entropy, vLLM, llama.cpp, PyTorch distributed, SQL Server

### TTT-E2E — Test-Time Training for Qwen3-4B

What if a model kept learning while it was being used — updating its weights from the very sequence it's reading?

- Implemented TTT-E2E for Qwen3-4B: second-order meta-gradients that optimize an initial weight state W₀ so that SGD steps on the input sequence actually improve next-token prediction in real-time.
- Explored the boundary between exact and approximate meta-learning. Exact second-order meta-gradients require Hessian-vector products that are incompatible with FlashAttention in PyTorch — a fundamental tension between computational efficiency and learning fidelity. FOMAML scales to 128K context; exact caps at ~1.5K.
- 4.6–6.1% perplexity improvement on PG19 (4K–32K context). The deeper insight: the boundary between training and inference is dissolving.
- **Technologies:** PyTorch, SDPA, FOMAML, meta-gradients, SwiGLU, RoPE

### Autonomous Equity Research Agent

How do you build an AI system that continuously monitors markets, evaluates opportunities, and curates a watchlist — without collapsing under its own context?

- Solved the context overflow problem in monolithic agent designs by decomposing into a microagent architecture: Topic Scanner → Symbol Assessor → Job Orchestrator. Each agent operates within its context budget, passing structured signals downstream.
- Integrated confidence-scored watchlist management, social sentiment analysis, and market cap filtering with full observability via WandB Weave.
- **Technologies:** OpenAI Agents SDK, MCP, SQL Server, SearXNG, WandB Weave, asyncio

### Jarvis — Multi-Agent Personal AI Runtime

Can a personal AI assistant maintain persistent identity, accumulate memory across conversations, and autonomously act on your behalf?

- Built an event-driven runtime with SQL-backed semantic memory and a layered persona system (SOUL, IDENTITY, BOOTSTRAP) that gives the AI a consistent identity across sessions.
- Heartbeat-driven automation enables the system to act proactively — executing trades via E\*TRADE, monitoring health data, managing linked accounts — without being prompted.
- **Technologies:** Python, SQLAlchemy, Signal CLI (JSON-RPC + SSE), systemd, OpenAI/Claude/Codex APIs

### Agent SDK — Reusable AI Agent Framework
- Built an extensible SDK for AI agents with automatic semantic context compaction at 80% token limits — solving the core challenge of keeping agents coherent in long-running tasks.
- Factory pattern enables specialist agent composition. Powers the equity research agent and other projects.
- **Technologies:** Python, OpenAI API, MCP, Playwright (browser automation via MCP), Logfire

### LlamaCraft — LLM Pre-Training from Scratch

Understanding LLMs means training them from scratch — not just fine-tuning.

- Pre-trained Llama 2 models from scratch on FineWeb-Edu. Explored the training dynamics of quantized training (float8, AdamW8bit, AdamW4bit, AdamWFp8) and how precision affects convergence.
- Published results on Weights & Biases and exported models to HuggingFace.
- **Technologies:** PyTorch, torchao, DDP/torchrun, FSDP, HuggingFace, WandB

### torchtitan — Custom Distributed Training Platform
- Extended PyTorch's torchtitan with checkpoint utilities for cross-world-size resumption, gradient accumulation memory optimization, and custom finance dataset integration (FNSPID).
- Built HuggingFace inference pipelines and model conversion tooling.
- **Technologies:** PyTorch, FSDP2, Tensor/Pipeline/Context Parallel, Float8, DCP

### MCP Compact — Context Optimization Proxy
- Open-source MCP proxy that applies LLM-based summarization to tool call responses, reducing context window consumption by up to 97% in agentic workflows.
- **Technologies:** Python, MCP SDK, OpenAI-compatible APIs, Docker

### Foundary — Neural Transaction Classifier
- Custom PyTorch Transformer encoder for financial transaction classification. Supports online learning — the model improves continuously as new transactions arrive, not just at training time.
- **Technologies:** PyTorch, FastAPI, Docker

### Ember Pulse — iOS Health Data Platform
- Full-stack health telemetry platform: Swift iOS app with HealthKit background sync and a FastAPI backend with WebAuthn passkey authentication. Exploring what becomes possible when personal health data is continuously streamed and queryable.
- **Technologies:** Swift, iOS 26 SDK, HealthKit, SwiftUI | Python, FastAPI, SQLAlchemy, WebAuthn/FIDO2, Docker

### Karpathy LLM Training Contributions
- **llama2.c** (14 commits): Added FineWeb/Dolphin dataset training, GPU data loading and buffering optimizations, hyperparameter tuning, WandB logging integration.
- **nanoGPT**: Added torch.compile, pin_memory, and async data loading optimizations.
- **llm.c**: Containerized C/CUDA training with Docker.
- **build-nanogpt**: Fixed PyTorch autocast device type for non-CUDA backends.

### RL Trading Experiments (Early Research)
- Built PPO reinforcement learning environments for automated stock trading — progressing from CartPole to custom trading environments with MLP and Transformer architectures. This early work led directly to the Qwen3-4B GRPO pipeline.
- **Technologies:** PyTorch, TorchRL, PPO, custom RL environments

## Professional Experience

### GuidedChoice — Reno, NV

**Chief Technology Officer** | May 2022 – Present

- Lead architecture, engineering, security, and infrastructure teams. Maintain 99.9%+ uptime across all services.
- Migrated from Docker Swarm to Kubernetes, enhancing scalability and resilience.
- **Technologies:** Java, Spring Cloud, Docker, Kubernetes, Azure, Kafka, SQL Server, Elasticsearch, React

**Product Manager, Architect, Lead Developer** | Sep 2017 – May 2022

- Architected and built a scalable microservices platform. Migrated legacy systems to modern architectures.
- Managed Oracle to MS SQL data migration. Mentored team on Spring Cloud and microservices patterns.
- **Technologies:** Java, Azure, Spring Cloud, Docker, SQL Server, React.js

### ARS National Service — Escondido, CA

**Enterprise Architect & Senior Developer** | Aug 2013 – Sep 2017

- Introduced Docker container architecture and Docker Swarm across data centers. Implemented WSO2 API Manager.
- Built REST APIs for legacy CRM, vendor integration systems, ETL tools, and document management systems.
- **Technologies:** Java, Spring Boot, C#, Docker, SQL Server, MongoDB

### CSUSM — San Marcos, CA

**iOS Developer** | Sep 2012 – Mar 2013
- Built augmented reality iOS app overlaying 3D objects in real-time camera views. (Objective-C, iOS)

**National Science Foundation** | Sep 2011 – Jun 2012
- Enhanced web application with Google Earth SDK for interactive geography learning. (PHP, MongoDB)

## Education

**Master of Science in Computer Science**
California State University, San Marcos — 2013

**Bachelor of Technology in Information Technology**
Easwari Engineering College, Anna University, Chennai, India — 2010
