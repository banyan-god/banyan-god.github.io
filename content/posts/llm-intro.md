---
title: "Embarking on My Journey into LLM"
date: 2024-12-27
description: "An introduction to my experiences and explorations in the world of Large Language Models."
author: "Sabareesh"
draft: false
summary: "Join a curious engineer’s quest into the fascinating world of Large Language Models (LLMs). From tinkering with GPUs to unraveling the mysteries of architectures like Llama2, this journey is filled with challenges, breakthroughs, and the relentless pursuit of understanding AI’s limitless potential."
categories:
  - Artificial Intelligence
  - Machine Learning
  - Technology
  - PreTraining

tags:
  - Large Language Models
  - LLM
  - Llama2
  - NanoGPT
  - Multi-GPU Training
  - AI Exploration
  - Model Optimization
  - PyTorch
  - AI Research
  - Deep Learning
  - Open-Source AI

---

# My Journey into Large Language Models (LLM)

My journey into Large Language Models (LLMs) began with the introduction of ChatGPT. Although AI advancements had been progressing for years, I only became aware of the field’s true potential after ChatGPT's breakthrough moment. This sparked my curiosity and led me to explore various models, ultimately motivating me to build a machine equipped with an NVIDIA 4090 GPU to experiment with open-source models.

Initially, I built a PC-grade setup with limited support for multiple PCIe lanes. Despite this limitation, it was sufficient to get started. I ran models like Mistral locally and experimented with their capabilities. My interest soon turned to multi-agent frameworks, leading me to experiment with Engineer GPT. I attempted to enable inter-model communication beyond token-based exchanges, which, while challenging, helped me realize the importance of understanding the inner workings of these models.

## Building the Foundations

I owe much of my foundational knowledge to Karpathy’s tutorials and Danielle Bourke’s "Learn PyTorch for Deep Learning in a Day." These resources introduced me to Python, PyTorch, and basic machine learning principles. From there, I started building simple models, beginning with linear layers and progressing to multilayer architectures featuring nonlinear activations like Tanh and ReLU.

After solving basic problems with deep neural networks, I moved on to constructing models using NanoGPT. Karpathy’s tutorials once again proved invaluable, offering clear guidance that made the learning process more accessible, even as a newcomer.

## Scaling Up

While running NanoGPT’s code, I learned a lot about setting up local machine learning environments. However, I quickly realized the limitations of relying on a single 4090 GPU, especially when working with larger models. The high demand for 4090 GPUs and the lack of PCIe lanes in my initial build prompted me to create a workstation from scratch. I chose a SuperMicro M12SWA-TF motherboard, an AMD Threadripper PRO 5955WX CPU, 128 GB of RAM, and a miner frame for the build.

![Distributed Training](/blog-images/ws-1.jpeg)

After acquiring a second 4090, I encountered a significant hurdle: NVIDIA had dropped P2P support for the 4xxx series. Thankfully, George Hotz’s kernel hack restored P2P functionality via messaging, allowing me to proceed with multi-GPU training.

This marked the beginning of a deep dive into multi-GPU training and its associated challenges. Resources from Hugging Face, Meta, and PyTorch helped me navigate this complex field. For anyone starting out, I recommend Hugging Face’s multi-GPU performance guide. Generally, Data Parallelism (DP) isn’t worth the effort—use Distributed Data Parallel (DDP) or the ZeRO family of techniques for better results.

### Distributed Training Insights

Training across multiple GPUs and nodes involves some ingenious engineering. In DDP, each GPU holds a complete copy of the model and trains in parallel. The updated weights are then merged on a primary GPU and redistributed to all others. Alternatively, if a model exceeds the capacity of a single GPU, you can split it horizontally, vertically, or both. ZeRO handles these scenarios effectively, enabling the training of massive models even with constrained hardware.

## Optimizing the Build

![4x 4090](/blog-images/ws-2.jpeg)

To further enhance my setup, I purchased two additional 4090 GPUs. Despite their 24 GB VRAM limitation, their tensor cores deliver exceptional performance, rivaling enterprise-grade GPUs like the A100 or H100. My final setup—featuring four 4090 GPUs—cost approximately $13,000 and offered a total of 96 GB VRAM.

Why not use the cloud? While cloud solutions are convenient, I wanted hands-on experience with inter-GPU communication and a dedicated platform for experiments that could run for weeks or months.

## Learning the Magic

Understanding the underlying mechanics of machine learning was critical. Two videos significantly enhanced my grasp of gradient descent and backpropagation:

- [Introduction to Neural Networks](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Deep Dive into Backpropagation](https://www.youtube.com/watch?v=q8SA3rM6ckI)

## Training with Llama2

I progressed to training models using the Llama2 architecture, which features RoPE embeddings. These embeddings facilitated larger batch sizes and improved scalability within the VRAM constraints. I also experimented with optimizers like `bnb.optim.AdamW8bit` to reduce memory usage and accelerate training.

Pretraining a model is just the beginning. Post-training processes like supervised fine-tuning and Reinforcement Learning with Human Feedback (RLHF) are essential for enhancing performance. I fine-tuned my models using datasets like Dolphin from Hugging Face, as well as newer datasets like FineWeb and FineWebEdu, which improved convergence rates.

## Challenges and Achievements

Training a 1B-parameter model was slow due to PCIe-4 bottlenecks and smaller batch sizes. I found that a 500M-parameter model was more resource-efficient. My experiments, which are available on [Weights & Biases](https://wandb.ai/banyan-t/llamac), include models trained on FineWeb 100B. Some of these are also published on [Hugging Face](https://huggingface.co/sabareesh88/fw14k).

## Exploring Reinforcement Learning

After gaining proficiency with LLMs, I turned my attention to reinforcement learning (RL). Starting with basic MLP models, I explored various RL algorithms. Using environments like OpenAI Gym and Gymnasium provided an excellent starting point, and I later integrated TorchRL for custom setups.

One critical lesson was the importance of carefully defining the environment. Poorly designed environments lead to models exploiting unintended shortcuts, deviating from the desired objectives. In many cases, MLPs outperformed transformers in RL tasks that didn’t require autoregressive behavior.

## Moving Forward

This journey is far from over. I am eager to explore new challenges, continue refining my skills, and contribute to the evolving landscape of LLMs and RL.

