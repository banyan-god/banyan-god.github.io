---

title: "All You Need to Build and Pretrain a 437.6M Parameter Model"
date: 2024-12-29T08:00:00
summary: "A detailed guide for building and pretraining a 437.6M parameter model with innovative methodologies and advanced optimizations."
author: "Sabareesh"
draft: true
categories:

- Artificial Intelligence
- Machine Learning
- Technology
- PreTraining

tags:

- Large Language Models
- LLM
- Llama2
- Multi-GPU Training
- Distributed Training
- PyTorch
- Model Optimization
- AI Research

---

### **Abstract**

Pretraining large language models is a critical step in building scalable and adaptable AI systems. This paper outlines the process of developing a 437.6M parameter Llama2-like model, focusing on architectural optimizations, tokenization strategies, and adaptive training methods. By leveraging open-source tools and custom datasets, the approach demonstrates how to achieve high efficiency in model convergence and performance. This guide is rooted in real-world experimentation, with key insights drawn from iterative training runs and lessons learned during the pretraining process.

---

### **1. Introduction**

Pretraining a language model from scratch provides unparalleled control over architecture, data representation, and domain-specific applications. This document emphasizes how a 437.6M parameter model can be built and trained effectively by combining innovations in model design and adaptive training practices. Drawing on experimentation with architectures like Llama2, the guide showcases methods to optimize pretraining for scalability and efficiency, making it accessible for researchers and practitioners alike.

An important distinction from Karpathy's Llama2.c is our approach to dataset preparation. While Llama2.c employs a unique strategy, our method prioritizes flexibility by making dataset switching seamless through the use of configurable preprocessing scripts. This is achieved by utilizing the `finewebedullama2-PreProcess.py` script, enabling rapid adaptation across datasets while ensuring densification. Such flexibility ensures researchers can experiment with various datasets with minimal configuration overhead.

---

### **2. Dataset Preparation**

**a. Data Sources**

- The primary dataset used is the **FineWebEdu Dataset** from Hugging Face (`HuggingFaceFW/fineweb-edu`).
  - The `sample-100BT` split was utilized for both training and validation.

**b. Cleaning and Preprocessing**

- Tokenization was performed using the `KoboldAI/llama2-tokenizer`, compatible with a 32k vocabulary.
- The following script snippet demonstrates the preprocessing pipeline:

```python
from transformers import AutoTokenizer

def preprocess_data(batch):
    tokenizer = AutoTokenizer.from_pretrained("KoboldAI/llama2-tokenizer")
    tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=1024)
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }
```

- **Densification of Dataset**: To maximize token utilization, sequences are split into fixed-length blocks and incomplete sequences are discarded. This process is critical because without densification, the model would train on padding or masking tokens, which could bias it to predict such tokens during inference. Initially, this issue caused the model to output masking tokens at the end of sequences. The solution was implementing the following custom collate function:

```python
def custom_collate_fn(batch, block_size):
    batch = [item for sublist in batch for item in sublist['input_ids']]

    input_ids, labels = [], []
    while len(batch) >= block_size + 1:
        current_input_ids = torch.tensor(batch[:block_size + 1])
        input_ids.append(current_input_ids[:-1])
        labels.append(current_input_ids[1:])
        batch = batch[block_size:]

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels)
    }
```

- Redundant columns were removed, and text was normalized to remove noise while preserving semantic richness.
- Data was split into 90/10 train-validation splits.

**c. Data Sharding**

- Tokenized data was sharded and stored efficiently to optimize I/O during training.

---

### **3. Model Architecture**

**Key Features of the 437.6M Parameter Model:**

- **Hidden Size**: 1024
- **Number of Layers**: 32
- **Attention Heads**: 32
- **Key-Value Heads**: 32
- **Sequence Length**: 1024

The model leverages RoPE embeddings to enhance scalability with long sequences and incorporates optimizations for gradient stability and memory efficiency. Below is an example of the model definition:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "banyan-god/LlamaCraft",
    config={
        "hidden_size": 1024,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "max_position_embeddings": 1024
    }
)
```

Advanced architectural choices make it robust for pretraining tasks involving large-scale datasets.

---

### **4. Training Configuration**

**a. Optimizer**

- **AdamW (bnb.optim.AdamW8bit)** for memory efficiency:
  - Weight decay: `1e-1`
  - Beta parameters: `beta1=0.9`, `beta2=0.95`

**b. Learning Rate Scheduling**

- Warmup for 1000 steps, followed by cosine decay.

**c. Mixed Precision**

- BF16 mixed precision enabled for optimal VRAM utilization.
- Example configuration for mixed precision training:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### **5. Results and Observations**

**a. Training Metrics**

- **Throughput**: To be verified from logged metrics during runs.
- **Perplexity Reduction**: Placeholder data removed pending actual results from training.

**b. Model Performance**

- RoPE embeddings facilitated consistent scaling with sequence lengths up to 1024 tokens.

**c. Challenges**

- Addressed vocabulary mismatches and convergence issues using dynamic learning rate adjustments.

---

### **References**

1. Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS 2017*. Retrieved from [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI GPT-2 Paper*. Retrieved from [https://openai.com/research/language-models-are-unsupervised-multitask-learners](https://openai.com/research/language-models-are-unsupervised-multitask-learners)
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." *GPT-3 Paper*. Retrieved from [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." *Meta AI Research*. Retrieved from [https://ai.meta.com/llama/](https://ai.meta.com/llama/)
5. Su, Y., et al. (2021). "Rotary Positional Embedding (RoPE) for Transformers." *Paper on RoPE*. Retrieved from [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
6. Karpathy, A. (2023). "Llama2.c: Minimalistic Llama2 Implementation." Retrieved from [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)
7. Hotz, G. (2023). "Tinygrad Open GPU Kernel Hack." Retrieved from [https://github.com/geohot/tinygrad](https://github.com/geohot/tinygrad)
8. Dettmers, T., et al. (2021). "8-bit Optimizers via BitsandBytes." *Efficient Optimization for Large Models*. Retrieved from [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
9. Sabareesh. (2024). "Training LLMs: Building Your Own Local Rig." Retrieved from [https://sabareesh.com/posts/llm-rig/](https://sabareesh.com/posts/llm-rig/)
10. Sabareesh. (2024). "Embarking on My Journey into LLM." Retrieved from [https://sabareesh.com/posts/llm-intro/](https://sabareesh.com/posts/llm-intro/)
11. Sabareesh. (2024). "Tracking Training Runs: Successes and Failures." Retrieved from [https://wandb.ai/banyan-t/llamac](https://wandb.ai/banyan-t/llamac)
12. Sabareesh. (2024). "LlamaCraft Repository for Code." Retrieved from [https://github.com/banyan-god/LlamaCraft](https://github.com/banyan-god/LlamaCraft)

