---
title: "Porting End-to-End Test-Time Training to PyTorch: From Paper to 128K Context on a Single GPU"
date: 2026-04-06
tags: ["deep-learning", "meta-learning", "long-context", "pytorch", "ttt"]
math: true
draft: false
---

We ported [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675) (TTT-E2E) from JAX to PyTorch, applied it to QWen3-4B, and trained at 128K context on a single RTX PRO 6000 Blackwell (96 GB). Along the way, we implemented both the paper's intended exact second-order meta-gradients and the first-order FOMAML approximation, and found that the difference between them reveals a fundamental tension in PyTorch's autograd: Flash Attention and second-order gradients are mutually exclusive.

Code: [github.com/banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3) | Paper: [arXiv:2512.23675](https://arxiv.org/abs/2512.23675) | Official JAX: [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e)

## What is TTT-E2E?

Transformers with full self-attention struggle at long context because the cost per token grows linearly with sequence length. The KV cache stores every past token's key and value, and attention must scan through all of them for each new token.

TTT-E2E takes a different approach: instead of storing the full history, **compress it into model weights**. The idea comes from a simple observation — pre-training also compresses massive data into weights via next-token prediction. Why not continue that process at test time?

The architecture has three components:

1. **Sliding Window Attention (SWA)** with a fixed window (8K tokens). This handles local, recent context — the model can directly attend to the last 8K tokens.

2. **Prime MLPs** in the last quarter of transformer layers. These are additional SwiGLU MLP layers whose weights get updated during test-time training. The original MLPs stay frozen, preserving pretrained knowledge.

3. **A meta-learned initialization** W₀ for the prime MLPs. At test time, an inner loop runs SGD on the prime MLPs using the input context as training data (next-token prediction loss). The outer loop at training time optimizes W₀ so that the model performs well *after* this inner-loop adaptation.

The key result from the paper: TTT-E2E scales with context length the same way as full attention (loss improves with more context), while having constant decode latency like an RNN. At 128K context on 3B models, it matches full attention quality while being 2.7x faster.

## The Meta-Learning Problem

The meta-learning is what separates TTT-E2E from simple dynamic evaluation. Without it, you're fine-tuning the model on the test input with no guarantee the starting weights are good for adaptation. The paper (Section 2.2) shows this clearly: TTT-naive (no meta-learning) barely improves over the baseline, while TTT-E2E (with meta-learning) nearly matches full attention.

The question is: **how do you meta-learn the initialization?**

The outer loss is the average of all next-token prediction losses across the inner-loop trajectory:

$$\mathcal{L}(W_0; X) = \frac{1}{T} \sum_{i=1}^{T/b} \sum_{t=(i-1)b+1}^{ib} \ell_t(W_{i-1})$$

where $W_i = W_{i-1} - \eta \frac{1}{b} \sum_t \nabla \ell_t(W_{i-1})$ is the mini-batch SGD update.

Optimizing this w.r.t. W₀ requires gradients of gradients, because the update rule itself contains a gradient operation.

## The Two Approaches

### Exact Second-Order (Paper's Approach)

With one inner step for clarity:

$$W_1 = W_0 - \eta \nabla L_0(W_0)$$

$$\frac{\partial L_1(W_1)}{\partial W_0} = \frac{\partial L_1}{\partial W_1} \cdot \left(I - \eta \frac{\partial^2 L_0}{\partial W_0^2}\right)$$

The Hessian term $\eta \frac{\partial^2 L_0}{\partial W_0^2}$ captures how the inner gradient direction changes as W₀ changes. This assumes vanilla SGD without momentum or clipping; the real implementation with clipped SGD has a more complex Jacobian, but the principle is the same.

In JAX, `jax.grad` naturally composes for higher-order derivatives. The [official implementation](https://github.com/test-time-training/e2e) uses `eqx.filter_value_and_grad` nested inside `scan_remat_chunk` for memory-bounded exact training.

### First-Order / FOMAML (Our Practical Approach)

FOMAML drops the Hessian by detaching the inner gradient computation from the outer graph:

$$\frac{\partial L_1(W_1)}{\partial W_0} \approx \frac{\partial L_1}{\partial W_1}$$

The Jacobian $\partial W_1 / \partial W_0$ is approximated as the identity matrix.

### Toy Example

With L(W) = (W - 3)², W₀ = 1.0, η = 0.1:

| | Exact | FOMAML |
|---|---|---|
| Outer gradient ∂L₁/∂W₀ | **-2.56** | **-3.20** |
| Jacobian ∂W₁/∂W₀ | 0.8 | 1.0 (identity) |

FOMAML overshoots by 25%. We verified both values on our implementation and they match the analytical solution.

## PyTorch Implementation

We implemented both modes in a single `ExactMetaLearner` class with a `meta_grad_mode` switch:

**Exact path** — no `.detach()` anywhere in the inner update chain:
- Prime params cloned with `.clone()` (graph-connected to original `nn.Parameter`)
- SWA KV state as immutable functional tuples (no in-place mutation)
- Inner SGD via `torch.autograd.grad(create_graph=True)`
- Segment checkpointing (`torch.utils.checkpoint`) over groups of inner steps
- Single `outer_loss.backward()` through the full trajectory
- **Math SDPA backend** (required — see below)

**FOMAML path** — detached inner loop:
- Prime params cloned with `.detach().clone().requires_grad_(True)`
- SWA KV cache detached after each chunk
- Per-chunk `.backward()` (graph freed immediately, O(1) memory)
- Flash/Efficient SDPA backend (fast)

### The Flash Attention Problem

A finding that isn't obvious from the paper: **PyTorch's Flash Attention and memory-efficient attention don't support second-order gradients.** Their custom backward kernels don't have derivative implementations:

```
RuntimeError: derivative for
aten::_scaled_dot_product_efficient_attention_backward
is not implemented
```

In exact mode, we must force the "math" backend (`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`), which materializes the full attention matrix. The official JAX implementation avoids this problem because JAX's XLA compiler generates Hessian-vector products automatically for any differentiable operation, including its attention kernels.

This is a fundamental asymmetry: **JAX's autograd is more flexible for higher-order gradients than PyTorch's custom CUDA kernels.**

## Measured Differences

All measurements on QWen3-4B base (4.0B params) + 672M added prime MLP params = 4.7B total. Single RTX PRO 6000 Blackwell, 96 GB VRAM, bf16 precision.

### Gradient Comparison (2 inner steps, 1025 tokens, mini_batch=512)

| | Exact | FOMAML |
|---|---|---|
| Peak GPU | 49.4 GB | 34.7 GB |
| All 27 prime params get gradient | Yes | Yes |

Max element-wise gradient difference: 0.0095. This is a single pair of runs on random tokens — not a downstream quality measurement, just a confirmation that the modes produce numerically different outer gradients as expected from the Hessian term.

### Memory Scaling

Both modes use the same SWA architecture (relative RoPE, fixed KV buffer). The difference is only in the gradient computation.

**FOMAML** (Flash SDPA, mini_batch=2048):

| Context | Inner Steps | Peak GPU | Throughput |
|---------|-------------|----------|------------|
| 8K | 4 | 28.4 GB | 2,961 tok/s |
| 32K | 16 | 41.6 GB | 2,714 tok/s |
| 64K | 32 | 52.1 GB | 2,055 tok/s |
| 128K | 64 | 75.9 GB | 1,370 tok/s |

**Exact** (Math SDPA, mini_batch=512 — different backend and chunk size):

| Context | Inner Steps | Peak GPU | Status |
|---------|-------------|----------|--------|
| 512 | 1 | 22.8 GB | OK |
| 1024 | 2 | 49.4 GB | OK |
| 1536 | 3 | 76.0 GB | OK |
| 2048 | 4 | >96 GB | OOM |

Note: these are not apples-to-apples comparisons. The exact mode uses a different SDPA backend (math vs flash) and a different chunk size (512 vs 2048), both of which affect memory independently of the gradient order. The combined effect is that FOMAML handles 128K context where exact mode can't even do 2K.

### Training Signal (FOMAML, 128K context)

Training on PG19 books with 4-sequence gradient accumulation (524K tokens per optimizer step):

- Loss: 2.87 → 2.49 over 20 optimizer steps
- Perplexity improvement from TTT at eval time:
  - 4K: 4.6% reduction vs no-TTT baseline
  - 8K: 4.6%
  - 16K: 4.6%
  - 32K: 6.1%

These are early training numbers (single run, 20 steps out of planned 2000). The trend of increasing improvement with context length is consistent with the paper's main claim.

## Architecture Details

Our implementation matches the [official JAX repo](https://github.com/test-time-training/e2e) on the key design decisions:

- **Sequential prime MLP**: attention → prime MLP (with own pre/post RMSNorm) → residual → original MLP. Not parallel.
- **Relative RoPE for suffix SWA**: positions [0, window+chunk) re-anchored every chunk. Verified against official `sw_causal_mask()`.
- **Fixed-size KV buffer**: stores raw (pre-RoPE) K/V, re-applies RoPE each chunk.
- **Prefix/suffix split**: first 27 layers run once on full sequence, last 9 layers unroll per chunk with inner-loop updates.
- **Only prime MLP weights updated during TTT**: attention, original MLPs, norms, embeddings, and LM head are frozen during the inner loop but receive outer-loop gradients.

Known differences from official: QWen3-4B base vs custom 3B, wider prime MLPs (9728 vs 5632), PG19 dataset vs Books3, FOMAML vs exact meta-gradients for long-context training.

## What We Learned

1. **Second-order meta-learning in PyTorch hits a wall** at Flash Attention's backward. JAX handles this transparently. Until PyTorch attention kernels support `create_graph=True`, exact meta-gradients at scale require the math backend, which nullifies the memory savings that make long context feasible.

2. **The prefix/suffix split is the key architectural insight.** Running prefix layers once and suffix layers per-chunk makes TTT-E2E practical. Without it, every inner step requires a full forward pass through all 36 layers.

3. **Relative RoPE in the suffix prevents position degradation.** At 128K+ context, absolute positions exceed the model's training range. Re-anchoring to [0, window+chunk) keeps positions bounded regardless of sequence length.

4. **Per-chunk backward with detached prefix** gives O(1) memory per inner step. This is the key to fitting 128K context on a single GPU — each chunk's graph is freed immediately after backward.

5. **The real bottleneck for 1M+ context** is the sequential chunk processing, not the gradient order. Even FOMAML at 1M tokens with 512 chunks of 2048 tokens would take ~6 minutes per optimizer step on our hardware.

---

*Implementation: [github.com/banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3)*
*Paper: [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675) — Tandon, Dalal, Li, Koceja, Rød, Buchanan, Wang, Leskovec, Koyejo, Hashimoto, Guestrin, McCaleb, Choi, Sun*
*Official JAX implementation: [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e)*
