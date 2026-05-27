# QLoRA Recipe Chef

A parameter-efficient recipe generation system built using QLoRA fine-tuning on Hugging Face’s RecipeNLG dataset and the Alibaba Cloud Qwen2.5-1.5B-Instruct model. The project demonstrates how low-rank adaptation and 4-bit quantization can train large language models efficiently while enabling controllable recipe generation through cuisine and dietary conditioning.

Based on the uploaded project walkthrough and presentation notes. 

---

## Project Overview

This project fine-tunes the Alibaba Cloud Qwen2.5-1.5B-Instruct model using QLoRA to generate structured cooking recipes from sparse ingredient lists.

The system supports:

* Cuisine-conditioned recipe generation
* Dietary preference conditioning
* Efficient 4-bit fine-tuning
* Gradio-based interactive demo
* Quantitative evaluation against zero-shot and few-shot baselines
* Memory-efficient training on consumer GPUs

The final adapter trains only ~0.6% of the model’s parameters while preserving the general capabilities of the base model. 

---

# Key Features

## QLoRA Fine-Tuning

The project uses:

* 4-bit NF4 quantization
* Double quantization
* LoRA adapters
* bf16 computation
* paged AdamW optimizer

This dramatically reduces memory usage while maintaining generation quality. 

---

## Cuisine & Diet Conditioning

The original RecipeNLG dataset does not include cuisine or dietary labels. This project introduces a custom rule-based labeling pipeline to generate:

### Cuisine Labels

Examples:

* Italian
* Mexican
* Indian
* Asian

### Dietary Labels

Examples:

* Omnivore
* Vegetarian
* Vegan

This allows controllable recipe generation using the same ingredient list. 

---

## Efficient Caching System

A Google Drive-backed caching layer allows:

* Recovery after Colab disconnects
* Reuse of trained adapters
* Persistent evaluation artifacts
* Faster reruns for collaborators

The first full pipeline run takes approximately 90 minutes, while cached reruns take around 5 minutes. 

---

# Model Architecture

## Base Model

* Model: Qwen2.5-1.5B-Instruct
* Parameters: ~1.55B
* Quantization: 4-bit NF4
* Compute precision: bf16

---

## LoRA Configuration

| Parameter            | Value  |
| -------------------- | ------ |
| Rank (r)             | 16     |
| Alpha                | 32     |
| Dropout              | 0.05   |
| Trainable Parameters | ~9.2M  |
| Total Parameters     | ~1.55B |
| Trainable Percentage | ~0.6%  |

LoRA adapters are attached to:

* q_proj
* k_proj
* v_proj
* o_proj
* gate_proj
* up_proj
* down_proj



---

# Dataset

## Source Dataset

Dataset: RecipeNLG (chat-formatted repack)

* Total recipes: ~2.23M
* Training subset: 20K
* Validation set: 500
* Test set: 500



---

## Dataset Processing

Each recipe is converted into a chat-style instruction format:

### System

Defines the assistant behavior

### User

Provides:

* Ingredients
* Cuisine preference
* Dietary preference

### Assistant

Generates:

* Recipe title
* Ingredients
* Instructions

Ingredient lists are randomly subsampled (3–8 ingredients) to better simulate real-world user prompts. 

---

# Training Configuration

| Hyperparameter       | Value            |
| -------------------- | ---------------- |
| Steps                | 400              |
| Effective Batch Size | 16               |
| Learning Rate        | 2e-4             |
| Optimizer            | paged_adamw_8bit |
| Precision            | bf16             |
| Max Sequence Length  | 512              |
| Scheduler            | Cosine           |
| Warmup               | 5%               |



---

# Evaluation

The fine-tuned model is evaluated against:

1. Base Qwen (zero-shot)
2. Few-shot prompting baseline
3. QLoRA fine-tuned adapter

## Metrics

* ROUGE-1
* ROUGE-2
* ROUGE-L
* BERTScore-F1
* Perplexity
* Ingredient Fidelity
* Knowledge Retention



---

# Major Findings

## QLoRA Memory Savings

The project demonstrates that QLoRA enables training large language models with significantly lower VRAM usage compared to full fine-tuning. 

---

## Strong Surface-Form Learning

ROUGE scores improve significantly after fine-tuning, indicating the model quickly learns:

* Recipe formatting
* Vocabulary patterns
* Structural conventions

Meanwhile, BERTScore changes are smaller, suggesting the base model already possessed cooking knowledge. 

---

## Improved Ingredient Fidelity

Fine-tuning substantially improves adherence to requested ingredients compared to few-shot prompting alone. 

---

## Knowledge Retention

The fine-tuned adapter preserves the general capabilities of the base model without catastrophic forgetting. 

---

# Gradio Demo

The project includes a Gradio interface with:

* Ingredient text input
* Cuisine dropdown
* Dietary preference dropdown
* Base vs fine-tuned comparison
* Example prompt gallery
* Ingredient fidelity scoring



---

# Repository Structure

```text
.
├── notebooks/
├── models/
├── figures/
├── rag_db/
├── app/
├── cached_results/
├── README.md
└── requirements.txt
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/hkbuttar/AI_Impact.git
cd AI_Impact
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Libraries

Main dependencies include:

* transformers==4.46.3
* peft==0.13.2
* trl==0.12.2
* bitsandbytes>=0.49.2
* accelerate
* datasets
* gradio
* torch



---

# Running the Project

## Launch Training Notebook

```bash
jupyter notebook
```

Open the main QLoRA notebook and execute sequentially.

---

## Launch Gradio App

```bash
python app.py
```

---

# Future Work

Potential extensions include:

* DPO / ORPO alignment training
* Multilingual recipe generation
* Vision-language integration for fridge-image recipe generation
* Larger-rank LoRA experiments
* Extended training on the full RecipeNLG dataset



---

# Contributors

* Harleen Buttar
* Team Members

---

# References

* QLoRA: Efficient Finetuning of Quantized LLMs
* LoRA: Low-Rank Adaptation of Large Language Models
* RecipeNLG Dataset
* Qwen2.5-1.5B-Instruct

---

# License

This project is intended for academic and research purposes.
