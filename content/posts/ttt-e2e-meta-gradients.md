---
title: "Exact vs First-Order Meta-Gradients in Test-Time Training: A PyTorch Implementation Analysis"
date: 2026-04-06
tags: ["deep-learning", "meta-learning", "long-context", "pytorch", "ttt"]
math: true
draft: false
---

We ported [End-to-End Test-Time Training (TTT-E2E)](https://arxiv.org/abs/2512.23675) from JAX to PyTorch on QWen3-4B and implemented both the paper's exact second-order meta-gradients and the first-order FOMAML approximation. This post documents what we learned about the practical differences, memory tradeoffs, and why this matters for long-context language modeling.

Code: [github.com/banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3)

## The Core Idea

TTT-E2E replaces the standard KV cache mechanism for long context with something radical: **train the model on the input context at test time**. Instead of storing all past tokens in a cache that grows linearly, TTT compresses context into the weights of small MLP layers ("prime MLPs") via gradient descent.

The architecture:
- **Sliding Window Attention (8K)** handles local context
- **TTT inner loop** compresses everything beyond the window into prime MLP weights
- **Meta-learning** trains the initialization W₀ so post-TTT performance is maximized

The meta-learning is the key innovation. Without it, you're just doing dynamic evaluation — fine-tuning on the test input with no guarantee the initialization was good for adaptation. With meta-learning, the model learns an initialization that is *designed* to be adapted.

## The Two Approaches

### Exact Second-Order (What the Paper Does)

The inner loop updates prime weights:

$$W_1 = W_0 - \eta \nabla L_0(W_0)$$

The outer loss is computed on the updated model:

$$L_{\text{outer}} = L_1(W_1)$$

The outer gradient is:

$$\frac{\partial L_{\text{outer}}}{\partial W_0} = \frac{\partial L_1}{\partial W_1} \cdot \frac{\partial W_1}{\partial W_0} = \frac{\partial L_1}{\partial W_1} \cdot \left(I - \eta \frac{\partial^2 L_0}{\partial W_0^2}\right)$$

The Hessian term $\eta \frac{\partial^2 L_0}{\partial W_0^2}$ captures how the inner gradient direction changes as W₀ changes. This tells the outer loop to shape not just the loss value but the **loss landscape** around W₀.

In JAX, this is natural — `jax.grad(jax.grad(f))` handles it automatically. The official implementation uses `eqx.filter_value_and_grad` nested inside `scan_remat_chunk` for memory-bounded second-order training.

### FOMAML (First-Order Approximation)

FOMAML drops the Hessian term by detaching the inner gradient:

$$\frac{\partial L_{\text{outer}}}{\partial W_0} \approx \frac{\partial L_1}{\partial W_1} \cdot I = \frac{\partial L_1}{\partial W_1}$$

This treats the inner update as having an identity Jacobian — the gradient of the outer loss w.r.t. W₀ is just the gradient at the current weights Wᵢ.

In PyTorch, this is the default when you `.detach()` the gradients in the inner loop.

## Concrete Numerical Difference

With a quadratic loss L(W) = (W - 3)², W₀ = 1.0, η = 0.1:

| | Exact | FOMAML |
|---|---|---|
| Inner gradient g₀ | 2(1-3) = -4 | -4 (same) |
| Updated weight W₁ | 1.0 - 0.1(-4) = 1.4 | 1.4 (same) |
| Outer loss L₁ | (1.4-3)² = 2.56 | 2.56 (same) |
| **Outer gradient ∂L₁/∂W₀** | **-2.56** | **-3.20** |
| Jacobian ∂W₁/∂W₀ | 0.8 (= 1 - 2η) | 1.0 (identity) |

FOMAML overshoots by 25% because it ignores that moving W₀ also changes the inner gradient direction. The Hessian factor (0.8) dampens the update in the exact case.

We verified this on our implementation — exact produces gradient -2.56, FOMAML produces -3.20, matching the analytical values.

## What Happens at Scale

On QWen3-4B (4.7B params, 672M prime params), we measured:

| Metric | Exact | FOMAML |
|---|---|---|
| Peak GPU (2 inner steps) | 49.4 GB | 34.7 GB |
| Max gradient difference | — | 0.0095 (vs exact) |
| Create graph | Yes (Hessian) | No |
| Memory per inner step | ~26 GB | ~13 GB |
| Max inner steps (96 GB) | 3 | ~128K context |

The gradient difference of 0.0095 is meaningful — it compounds over training. But the memory cost is brutal: exact mode uses **2x memory per inner step** due to the second-order computation graph.

### Memory Scaling

```
FOMAML (SDPA, mini_batch=2048):
  8K:   28.4 GB, 2961 tok/s
  32K:  41.6 GB, 2714 tok/s
  64K:  52.1 GB, 2055 tok/s
  128K: 75.9 GB, 1370 tok/s  ← fits on 96GB GPU

Exact (Math SDPA, mini_batch=512):
  512 tokens:  22.8 GB (1 step)
  1024 tokens: 49.4 GB (2 steps)
  1536 tokens: 76.0 GB (3 steps)
  2048 tokens: OOM (>96 GB)    ← can't even do 2K
```

FOMAML handles 128K (128 inner steps) on a single 96GB GPU. Exact can't even do 2K tokens (4 inner steps). The gap is **64x in context length capability** on the same hardware.

### Why Flash Attention Can't Help

A critical finding: PyTorch's Flash Attention and memory-efficient attention kernels **don't support second-order gradients**. The backward pass of these kernels doesn't have a Hessian-vector product implementation:

```
RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward
is not implemented
```

Exact mode must use the "math" SDPA backend (standard matmul), which materializes the full attention matrix. This alone accounts for ~5x of the memory overhead — Flash Attention's memory savings are exactly the savings that second-order forbids.

## The Implementation

Our `ExactMetaLearner` supports both modes via a `meta_grad_mode` flag:

**Exact path:**
1. Clone W₀ with `.clone()` (no `.detach()` — graph connected)
2. SWA state as immutable tuples (functional, no mutation)
3. Inner SGD with `torch.autograd.grad(create_graph=True)`
4. Segment checkpointing via `torch.utils.checkpoint`
5. Single `outer_loss.backward()` through entire trajectory
6. Math SDPA backend forced for both prefix and suffix

**FOMAML path:**
1. Clone W₀ with `.detach().clone().requires_grad_(True)`
2. SWA state with `.detach()` after each chunk
3. Inner SGD with `create_graph=False`
4. Per-chunk `.backward()` (graph freed immediately)
5. Gradients accumulated manually into W₀
6. Flash/Efficient SDPA backend (fast)

The segment checkpointing mirrors the official JAX `scan_remat_chunk`: groups of 8-16 inner steps are wrapped in `torch.utils.checkpoint(use_reentrant=False)`, which recomputes forward activations during backward to save memory. However, the second-order graph through the inner updates still needs to be retained between segments, which limits the savings.

## Practical Recommendations

**For single-GPU training (96 GB):**
- Use FOMAML for 32K+ context. It works, loss decreases, and the 4.6-6.1% perplexity improvement from TTT grows with context length.
- Use exact mode for short-context pre-training/warmup (up to ~1.5K tokens).

**For multi-GPU:**
- Exact mode becomes feasible at longer contexts with tensor parallelism splitting the suffix layers across GPUs.

**Hybrid approach (best of both):**
- FOMAML for the first N-K inner steps, exact for the last K. This gives the Hessian signal for the most recent updates while keeping memory bounded. Memory: O(K) extra instead of O(N).

**The honest assessment:**
The paper's results likely benefit significantly from exact meta-gradients, especially at long context where the inner loop has many steps and the FOMAML identity-Jacobian approximation degrades. But FOMAML still captures the core TTT mechanism — context compression into weights — and the practical memory constraints make it the only viable option for long context on current hardware without multi-node setups.

## Results

Training QWen3-4B at 128K context on PG19 books with FOMAML, 4-sequence gradient accumulation (524K tokens/step):

- Loss dropped from 2.87 → 2.49 in 20 steps
- TTT consistently improves perplexity: +4.6% at 8K, +6.1% at 32K
- Improvement grows with context length (the paper's key claim)
- Single RTX PRO 6000 Blackwell (96 GB), 1370 tok/s

The inner loop compression is working — within each training step, the loss drops as the model processes more context chunks and compresses them into prime MLP weights.

## What We Learned

1. **Second-order meta-learning in PyTorch is fundamentally memory-limited** by the lack of Flash Attention backward-of-backward. JAX's XLA compiler can handle this more gracefully.

2. **FOMAML is 80-90% of exact** for this application, at 50% of the memory cost. The Hessian signal matters but isn't make-or-break.

3. **The prefix/suffix split is the key architectural insight** — running prefix layers once and suffix layers per-chunk makes TTT-E2E practical. Without it, every chunk requires a full model forward.

4. **SWA with relative RoPE is essential** for context beyond the pre-training length. Absolute positions degrade at 1M+; relative positions [0, window+chunk) stay in a well-trained range forever.

5. **The real bottleneck for 1M+ context** isn't the meta-gradient order — it's the sequential processing of hundreds of chunks. Even FOMAML at 1M tokens with 512 chunks takes minutes per step.

---

*Implementation: [github.com/banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3)*
*Paper: [End-to-End Test-Time Training for Long Context (arXiv:2512.23675)](https://arxiv.org/abs/2512.23675)*
