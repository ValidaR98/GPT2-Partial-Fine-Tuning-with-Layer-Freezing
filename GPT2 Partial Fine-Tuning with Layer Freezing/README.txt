#  GPT2 Partial Fine-Tuning with Layer Freezing  
This project explores **selective fine-tuning of GPT-2** by freezing internal transformer layers and training only a subset of parameters. The goal is to understand how GPT-2 adapts when only specific components are allowed to update, using *Gulliver’s Travels* as the training corpus.

---

## Project Overview

This repository demonstrates:

- How to freeze GPT-2 transformer blocks for efficient lightweight fine-tuning  
- How to train only selected layers while monitoring weight updates  
- How next-token prediction behaves under partial fine-tuning  
- How training loss evolves over time  
- How to verify which layers updated and which remained frozen  

This technique is widely used in:
- Low-resource fine-tuning  
- Domain adaptation  
- Efficient LLM training setups  
- Research on parameter-efficient tuning (PET, LoRA, adapters)

---

##  How the Model is Fine-Tuned

The project:

1. Loads the pretrained GPT-2 model  
2. Freezes all transformer layers (`.h.` layers)  
3. Leaves final normalization and LM head **trainable**  
4. Samples random sequences from *Gulliver’s Travels*  
5. Performs next-token prediction training  
6. Tracks the evolution of frozen vs. trainable weights  

Only **non-transformer layers** update, dramatically reducing compute cost.

---

##  Dataset

Text Source: *Gulliver’s Travels* — Project Gutenberg  
URL: https://www.gutenberg.org/cache/epub/829/pg829.txt

The text is automatically downloaded and tokenized with `GPT2Tokenizer`.

---

## Training Configuration

- Model: `gpt2` (HuggingFace Transformers)
- Sequence length: 256  
- Batch size: 16  
- Optimizer: AdamW  
- Learning rate: 5e-5  
- Weight decay: 0.01  
- Training iterations: 123 samples  

---

##  Results & Analysis

The project plots training loss across iterations.  
After training, the script prints:

- Norm difference in **frozen layer weights** (should remain ~0)  
- Norm difference in **trainable layer weights** (should be >0)  

This confirms which layers updated during fine-tuning.

---

##  How to Run

Install dependencies:

```bash
pip install torch transformers matplotlib requests numpy
