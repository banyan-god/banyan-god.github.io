# Sabareesh Subramani

[https://sabareesh.com](https://sabareesh.com) | [LinkedIn](https://www.linkedin.com/in/sabaree/) | [hello@sabareesh.com](mailto:hello@sabareesh.com)

## Summary

Machine learning engineer building end-to-end AI trading systems — from pre-training LLMs from scratch to fine-tuning Qwen3-4B on proprietary financial datasets and training trading decision models with GRPO reinforcement learning. Operates custom GPU infrastructure (4x RTX 4090 + 2x RTX PRO 6000 Blackwell) for continuous experimentation. Developed a multi-stage waterfall pipeline (equity knowledge injection → instruct alignment → stock prediction → trade execution) achieving 100% format validity and +9.4% portfolio return. Designed microagent architectures for autonomous equity research and a multi-agent personal AI runtime. 14 years of production software engineering experience as CTO, with deep expertise in scalable systems, Kubernetes, and cloud infrastructure.

## Technical Skills

- **LLM Training:** PyTorch, torchtune, torchao (float8/4-bit quantized training), TRL, NeMo-RL (GRPO/DAPO), DDP, FSDP/FSDP2, torchrun, vLLM, llama.cpp, Cut Cross-Entropy
- **Model Architectures:** Qwen3-4B, Llama 2/3, DeepSeek R1, GPT-2/NanoGPT, custom Transformer encoders
- **RL for LLMs:** GRPO, DAPO, PPO, TorchRL, volatility-normalized reward functions, counterfactual opportunity regret, hold-penalty scheduling, reward shaping for trading
- **Agent Systems:** OpenAI Agents SDK, MCP (Model Context Protocol), multi-agent orchestration, context compaction, semantic memory, WandB Weave observability
- **ML Infrastructure:** WandB, HuggingFace Hub, CUDA, Ray, multi-GPU training (4x RTX 4090, 2x RTX PRO 6000 Blackwell), checkpoint management, vLLM evaluation pipelines
- **Languages:** Python, Java, Swift, C, C++, TypeScript, C#
- **Backend & Infrastructure:** FastAPI, Spring Boot/Cloud, SQLAlchemy, Docker, Kubernetes, Azure, AWS, SQL Server, Kafka, Elasticsearch

## AI & ML Projects

### Qwen3-4B Financial Trading Model — SFT + RL Pipeline

Multi-stage waterfall fine-tuning and reinforcement learning pipeline for training an autonomous equity trading decision model on Qwen3-4B.

**Supervised Fine-Tuning (5-stage waterfall):**
- Built dataset pipelines exporting equity reports (5,176 reports), stock prediction data (103K records across 94 symbols), and trading decisions from SQL Server into structured JSONL.
- Trained through 5 stages: equity knowledge injection, Alpaca instruct alignment, stock prediction (achieved ~5-6% MAPE), stateful trading decisions, and distributed training.
- Developed weighted CCE loss (using Apple's Cut Cross-Entropy — 4.7x faster, 3.2x less memory) with conditional per-field, per-sample weights. Enter-critical fields (decision, side, stop_loss) weighted 4-8x; hold-null fields suppressed.
- Diagnosed and solved hold-collapse failure mode where model learned trivial all-hold policy due to class imbalance. Identified epoch12 checkpoint as optimal recovery base over epoch15/18.

**Key result — best checkpoint (v2 weighted CCE):** 100% format validity, 13.3% enter rate, 136 trades, $109,428 final equity (+9.4% return) on 1024-record portfolio evaluation.

**Reinforcement Learning (current phase):**
- Pivoting from SFT to GRPO after SFT hit its ceiling — model imitates oracle at 80% match rate but captures <10% of oracle alpha (oracle uses future information).
- Designed volatility-normalized, bounded reward function [-2, +2]: `0.8 * pnl_score - 0.2 * dd_penalty` using `tanh(return / sigma_h)` for enters; counterfactual opportunity regret for holds (penalizes missing >1.25 sigma moves).
- Research survey covering Trading-R1 (Sharpe 2.72 on NVDA), FLAG-Trader (LLM as RL policy with LoRA + PPO), Alpha-R1, HCAPO (hindsight critic), and DAPO at FinRL Contest 2025 (230% cumulative return).
- Designed "Living Trader" architecture with 4 loops: real-time inference, daily experience collection, weekly RL training, continuous self-research.

**Scientific methodology:** Maintained experiment journal with 13+ dated entries, formal scientific reports, per-experiment leaderboards with 12 ranked runs, defined alpha thesis with explicit falsification criteria, and automated phase gates for SFT → RL → regime adaptation → paper deployment.

- **Technologies:** torchtune, TRL, NeMo-RL (Ray + vLLM), Cut Cross-Entropy, vLLM, llama.cpp, PyTorch distributed, SQL Server
- **Hardware:** 2x NVIDIA RTX PRO 6000 Blackwell (96GB each), 4x RTX 4090

### Autonomous Equity Research Agent
- Designed and built an autonomous equity research and watchlist curation system using the OpenAI Agents SDK with MCP browser automation.
- Implemented a microagent architecture (Topic Scanner → Symbol Assessor → Job Orchestrator) to solve context overflow problems in monolithic agent designs.
- Integrated confidence-scored watchlist management, social sentiment analysis, and market cap filtering with full observability via WandB Weave.
- **Technologies:** OpenAI Agents SDK, MCP, SQL Server, SearXNG, WandB Weave, asyncio

### Jarvis — Multi-Agent Personal AI Runtime
- Built a personal AI assistant runtime with event-driven Signal messaging, SQL-backed semantic memory, and a layered persona/identity system (SOUL, IDENTITY, BOOTSTRAP).
- Features heartbeat-driven periodic automation, local skill execution (E\*TRADE integration, linked-account transactions, health data explorer), and multi-provider AI support (Claude, OpenAI, Codex).
- Deployed as systemd services with Signal CLI daemon for real-time communication.
- **Technologies:** Python, SQLAlchemy, Signal CLI (JSON-RPC + SSE), systemd, OpenAI/Claude/Codex APIs

### Agent SDK — Reusable AI Agent Framework
- Built a clean, extensible SDK for AI agents with tool-calling capabilities, MCP integration, and automatic semantic context compaction at 80% token limits.
- Factory pattern enables multi-agent systems with specialist agent composition. Powers the equity research agent and other projects.
- **Technologies:** Python, OpenAI API, MCP, Playwright (browser automation via MCP), Logfire

### LlamaCraft — LLM Pre-Training from Scratch
- Pre-trained Llama 2 models from scratch on FineWeb-Edu using a custom 4x RTX 4090 workstation.
- Implemented quantized training with torchao (float8, AdamW8bit, AdamW4bit, AdamWFp8) and CPU offloading. Trained with DDP via torchrun.
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
- Production text classification microservice with a custom PyTorch Transformer encoder for financial transactions. Supports online learning, batch inference, and multi-backend (CUDA/MPS/CPU) serving.
- **Technologies:** PyTorch, FastAPI, Docker

### Ember Pulse — iOS Health Data Platform
- Built the full stack: Swift iOS app collecting HealthKit telemetry with background sync (HKAnchoredObjectQuery, HKObserverQuery), and a FastAPI Python backend with WebAuthn passkey authentication and JWT rotation.
- **Technologies:** Swift, iOS 26 SDK, HealthKit, SwiftUI | Python, FastAPI, SQLAlchemy, WebAuthn/FIDO2, Docker

### Karpathy LLM Training Contributions
- **llama2.c** (14 commits): Added FineWeb/Dolphin dataset training, GPU data loading and buffering optimizations, hyperparameter tuning for 4090, WandB logging integration.
- **nanoGPT**: Added torch.compile, pin_memory, and async data loading optimizations.
- **llm.c**: Containerized C/CUDA training with Docker.
- **build-nanogpt**: Fixed PyTorch autocast device type for non-CUDA backends.

### RL Trading Experiments (Early Research)
- Built PPO reinforcement learning environments for automated stock trading using custom PyTorch models and TorchRL. Progressed from CartPole to custom stock trading environments with MLP and Transformer architectures.
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
