---
title: "Test-Time Training for LLMs: Models Should Keep Learning After Deployment"
date: 2026-04-06
tags: ["deep-learning", "meta-learning", "long-context", "pytorch", "ttt", "online-learning"]
math: true
draft: false
---

Pretraining taught us that neural networks can compress massive amounts of data into weights. But once we deploy an LLM, we usually stop that process completely. The model becomes frozen — it reads new inputs but never learns from them.

Test-time training asks a more ambitious question: **what if the model kept learning while it was being used?**

[TTT-E2E](https://arxiv.org/abs/2512.23675) is one practical answer. It lets a language model adapt its weights online from the very sequence it is reading. One consequence is dramatically stronger long-context behavior — but the deeper insight is that inference and learning don't have to be separate phases.

We ported TTT-E2E from JAX to PyTorch, applied it to QWen3-4B, and trained at 128K context on a single GPU. This post documents what we learned.

Code: [github.com/banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3) | Paper: [arXiv:2512.23675](https://arxiv.org/abs/2512.23675) | Official JAX: [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e)

## Why Frozen Models Are the Wrong Abstraction

Consider your experience reading this post. You don't process each sentence in isolation — you build up context, adjust your understanding, and use earlier information to interpret what comes later. Your "weights" are changing as you read.

Current LLMs don't do this. They have a fixed set of weights and a growing KV cache. The cache stores verbatim key-value pairs from past tokens, but the model's understanding — its weights — never changes during inference. This means:

- The model can't compress earlier context into a more useful form
- The KV cache grows linearly with context, making long sequences expensive
- Each input is processed with the same static model, regardless of what it contains

Retrieval-augmented generation and larger context windows are the standard responses. But they don't address the fundamental issue: **the model isn't learning from what it reads.**

## What Test-Time Training Is

Test-time training (TTT) makes inference include learning. The model updates itself during inference using the current input as training data:

1. The model sees a chunk of input tokens
2. It computes a next-token prediction loss on those tokens
3. It takes a gradient step to update a subset of its weights
4. It continues processing with the updated weights

This is online learning in the classical sense — the model adapts to the data distribution it encounters at test time. The key difference from offline fine-tuning is that it happens in real-time, on a per-input basis, and the updates are designed to be temporary (the model resets for the next input).

The concept has a long history: dynamic evaluation in NLP, test-time augmentation in vision, and meta-learning frameworks like MAML. TTT-E2E brings it to scale with modern transformers.

## TTT-E2E: One Concrete Implementation

[TTT-E2E](https://arxiv.org/abs/2512.23675) (Tandon, Dalal, Li, Koceja, Rød, Buchanan, Wang, Leskovec, Koyejo, Hashimoto, Guestrin, McCaleb, Choi, Sun) is an architecture designed to make test-time training practical for language models. It has three components:

**Sliding Window Attention (8K tokens)** handles local context. The model can directly attend to recent tokens within the window, just like a standard transformer with a fixed-size attention span.

**Prime MLPs** are additional SwiGLU layers added to the last quarter of transformer blocks. Their weights are updated during test-time training via SGD on next-token prediction loss. The original MLPs stay frozen, preserving pretrained knowledge. Think of it as: the original MLP stores what the model learned during pretraining, the prime MLP stores what it's learning right now from this specific input.

**Meta-learned initialization (W₀)** is the critical ingredient. At training time, the model doesn't just learn good weights — it learns an initialization for the prime MLPs that is *designed to be adapted*. The training objective is: "optimize W₀ so that after running TTT gradient steps on an input sequence, the model performs well on predicting the next tokens." This requires differentiating through the TTT update rule — gradients of gradients.

The model's forward pass on a long sequence looks like this:

```
For each chunk of 2048 tokens in the 128K context:
  1. Run attention + prime MLP + original MLP
  2. Compute next-token prediction loss
  3. SGD step on prime MLP weights only
  → Prime MLPs now "remember" something about this chunk

After processing all chunks:
  The prime MLPs contain a compressed representation of the full context
  → Decode new tokens using the adapted model
```

The model is not just reading the context — it is compressing parts of that context into weights as it goes.

## Why Long-Context Improves

Long-context performance is a natural consequence of online adaptation, not the primary goal. Here's why:

Standard transformers with full attention can represent long-range dependencies, but the cost per token grows linearly with context. Sliding window attention is cheap but can only "see" the last 8K tokens.

TTT-E2E gives the model a second mechanism for carrying forward information besides the attention KV cache: **adapted weights**. The prime MLPs accumulate knowledge from all past chunks, not just the ones within the attention window. Information from token 1 can influence prediction at token 128K through the weight updates, even though it's long outside the attention window.

The paper shows that TTT-E2E scales with context length the same way as full attention — loss improves as context grows — while having constant decode latency like an RNN.

## What We Implemented

We ported the full TTT-E2E architecture to PyTorch on QWen3-4B (4.0B base params + 672M added prime MLP params):

- **Prefix/suffix split**: the first 27 layers run once on the full sequence; the last 9 layers (with prime MLPs) run per-chunk with inner-loop updates
- **Sliding window attention with relative RoPE**: positions re-anchored to [0, window+chunk) every chunk, matching the [official JAX implementation](https://github.com/test-time-training/e2e). This keeps RoPE positions bounded regardless of sequence length.
- **SDPA-accelerated attention**: 5.6x faster than manual matmul for suffix attention
- **Per-chunk backward with O(1) memory**: each chunk's graph is freed immediately after backward, enabling arbitrarily long context on a single GPU
- **Gradient accumulation**: 4 sequences per optimizer step (524K tokens/step)
- **Both meta-gradient modes**: exact second-order (paper-faithful) and first-order FOMAML (practical)

## The Main Engineering Tradeoff

The paper's meta-learning requires differentiating through the TTT update rule (gradients of gradients). In JAX this is natural. In PyTorch, we found a fundamental constraint:

**Flash Attention and second-order gradients are mutually exclusive.**

PyTorch's Flash Attention kernel doesn't have a Hessian-vector product implementation. When you need `create_graph=True` for the inner-loop gradients (to get the exact meta-gradient), you must fall back to the "math" SDPA backend, which materializes the full attention matrix. This erases Flash Attention's memory savings — the exact savings you need for long context.

The practical consequence:

| Mode | Max context (96 GB GPU) | Throughput |
|---|---|---|
| Exact second-order | ~1.5K tokens | — |
| FOMAML (first-order) | **128K tokens** | 1,370 tok/s |

We implemented both modes with a single `meta_grad_mode` switch. For long-context training, FOMAML is the only viable path on current hardware. The exact mode is useful for short-context pre-training or verification.

The gradient difference between the modes is real and measurable. On a toy quadratic example, exact produces gradient -2.56 where FOMAML gives -3.20 (the Hessian factor dampens the update). On the full 4.7B model, the element-wise gradient difference is ~0.01, which compounds over training.

## Results

Training at 128K context on PG19 (Project Gutenberg books), single RTX PRO 6000 Blackwell (96 GB):

**Training loss** (FOMAML, 4-sequence accumulation, 524K tokens/step):
- Step 2: 2.97
- Step 20: 2.49 (Δ=-0.40)
- Continuing to decrease

**Perplexity improvement from TTT** (evaluated on a step 60 checkpoint):

| Context | Without TTT | With TTT | Improvement |
|---|---|---|---|
| 4K | 12.0 | 11.4 | 4.6% |
| 8K | 12.6 | 12.0 | 4.6% |
| 16K | 10.1 | 9.6 | 4.6% |
| 32K | 15.2 | 14.2 | 6.1% |

The improvement grows with context length. This is consistent with the paper's central claim and with the online-learning interpretation: the more context the model can learn from, the more benefit it gets from adaptation.

These are early training numbers from a single run — we include them as directional evidence, not final results.

## What This Says About Future LLMs

Long context is one application of test-time training. The larger idea is that deployed models should be adaptive, not static.

Consider:
- A coding assistant that adapts to your codebase conventions as it reads your files
- A medical model that adjusts to a patient's specific terminology and history
- A reasoning model that fine-tunes its internal representations on each problem

These are all instances of the same principle: **the model should keep learning from its input, not just process it.**

TTT-E2E demonstrates that this is architecturally feasible at scale. The mechanism — compressing context into fast weights via gradient descent — is one approach. Others may emerge. But the direction seems clear: the boundary between training and inference is dissolving.

---

*Implementation: [github.com/banyan-god/ttt-e2e-qwen3](https://github.com/banyan-god/ttt-e2e-qwen3)*
*Paper: [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675)*
*Official JAX: [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e)*
