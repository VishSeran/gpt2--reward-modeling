# reward-modeling-with-trl

> A hands-on lab for training a reward model using Hugging Face's TRL library, GPT-2, and RLHF-style pairwise preference data.

---

## Overview

This project walks through the end-to-end process of building and evaluating a **reward model** — a key component in Reinforcement Learning from Human Feedback (RLHF) pipelines. Using the `trl` (Transformer Reinforcement Learning) library from Hugging Face, the notebook fine-tunes a GPT-2 model for sequence classification on a synthetic pairwise preference dataset, teaching it to score responses and prefer higher-quality outputs.

This is particularly relevant for teams working on aligning large language models (LLMs) to produce responses that meet specific quality standards, follow complex instructions, or reflect organizational values.

---

## What You'll Learn

- The concept of reward modeling and its role in RLHF
- How to load and explore the [`Dahoas/synthetic-instruct-gptj-pairwise`](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise) dataset
- How to set up and configure GPT-2 for sequence classification using `GPT2ForSequenceClassification`
- How to apply **LoRA (Low-Rank Adaptation)** via the `peft` library for parameter-efficient fine-tuning
- How to preprocess and tokenize pairwise (chosen/rejected) response data
- How to train a reward model using `RewardTrainer` from `trl`
- How to evaluate the model using pairwise logit comparison and accuracy metrics

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `torch` | 2.3.1 | Deep learning framework |
| `transformers` | 4.43.4 | GPT-2 model & tokenizer |
| `trl` | 0.11 | `RewardTrainer` for RLHF |
| `peft` | 0.14.0 | LoRA configuration |
| `datasets` | 3.2.0 | Hugging Face dataset loading |
| `bitsandbytes` | 0.43.1 | Quantization support |
| `matplotlib` | 3.10.0 | Visualization |

---

## Dataset

**[Dahoas/synthetic-instruct-gptj-pairwise](https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise)**

A synthetic dataset of prompt-response pairs where each sample contains:
- `prompt` — An instruction or question
- `chosen` — The preferred (higher quality) response
- `rejected` — The less preferred response

This structure is ideal for training reward models that can rank and score responses.

---

## Project Structure

```
RewardTrainer-v1.ipynb       # Main lab notebook
```

### Notebook Sections

1. **Setup** — Install dependencies, import libraries, define helper utilities
2. **Dataset** — Load and explore the pairwise preference dataset
3. **Model & Tokenizer Setup** — Initialize GPT-2 with sequence classification head
4. **Preprocessing** — Combine prompts with responses, filter by length, tokenize
5. **LoRA Configuration** — Set up parameter-efficient fine-tuning
6. **Training** — Configure `TrainingArguments` and run `RewardTrainer`
7. **Evaluation** — Score responses with logits and measure pairwise accuracy
8. **Exercise** — Evaluate model preference accuracy on a different data subset

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/reward-modeling-with-trl.git
cd reward-modeling-with-trl
```

### 2. Install dependencies

```bash
pip install torch==2.3.1 datasets==3.2.0 trl==0.11 huggingface_hub==0.36.0 \
    transformers==4.43.4 peft==0.14.0 nltk==3.9.1 rouge_score==0.1.2 \
    bitsandbytes==0.43.1 matplotlib==3.10.0 rich==13.7.1
```

### 3. Launch the notebook

```bash
jupyter notebook RewardTrainer-v1.ipynb
```

> **Note:** Restart the kernel after installing dependencies, as prompted in the notebook.

---

## How It Works

The reward model is trained to assign a scalar score to each response. During evaluation, given a `chosen` and a `rejected` response to the same prompt, the model's logits are compared:

```python
if logit_chosen > logit_rejected:
    # Model correctly prefers the better response ✓
```

Accuracy is calculated as the percentage of pairs where the model correctly ranks the chosen response above the rejected one.

---

## Key Concepts

**Reward Modeling** — Training a model to predict a scalar reward signal that reflects human preferences, used as a proxy for human feedback in RLHF.

**LoRA (Low-Rank Adaptation)** — A parameter-efficient fine-tuning technique that injects small trainable matrices into the model, significantly reducing the number of trainable parameters.

**RewardTrainer** — A specialized trainer from Hugging Face's `trl` library designed for training reward models on pairwise preference data.

