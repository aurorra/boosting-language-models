# Boosting Real-Word Error Detection

Code accompanying the paper:

**Corina Masanti, Hans-Friedrich Witschel, and Kaspar Riesen (2026).  
*Enhancing language models with boosting and targeted fine-tuning for real-word error detection.*  
Natural Language Processing Journal, 14, 100202.**

Paper: https://doi.org/10.1016/j.nlp.2026.100202

Synthetic dataset: https://huggingface.co/datasets/aurorra/synthetic-real-word-errors

## Overview

This repository contains the code used to investigate German real-word error detection with transformer-based language models.

The experiments compare:

- standard fine-tuning,
- a boosting-inspired training strategy that iteratively adds synthetic examples for error patterns missed on the validation set,
- random selection of additional synthetic data as an ablation,
- targeted fine-tuning based on remaining error patterns, and
- evaluation on balanced and realistic error distributions.

The models used in the paper are:

- `bert-base-multilingual-cased` (mBERT),
- `DiscoResearch/Llama3-German-8B`, and
- `LeoLM/leo-mistral-hessianai-7b`.

Targeted fine-tuning was evaluated with mBERT only; therefore, no LoRA targeted-fine-tuning script is included.

## Repository Structure

```text
boosting-language-models/
├── baseline/
│   ├── train_mbert.py
│   └── train_lora.py
├── boosting/
│   ├── boost_mbert.py
│   └── boost_lora.py
├── evaluation/
│   └── inference.py
├── random_selection/
│   ├── random_selection_mbert.py
│   └── random_selection_lora.py
├── targeted_finetuning/
│   └── targeted_finetuning_mbert.py
├── README.md
└── LICENSE
```

## Data

The synthetic German real-word error dataset is available on Hugging Face:

https://huggingface.co/datasets/aurorra/synthetic-real-word-errors

It contains synthetic examples for:

- capitalization errors,
- case errors,
- verb errors, and
- a combined real-word error setting.

The public Hugging Face release is a merged release of the synthetic data. The training scripts in this repository currently reflect the original experimental setup and expect local `training.csv`, `validation.csv`, and `test.csv` files.

The real-world proofreading corpus used for evaluation in the paper is not included in this repository.

## Methods

### Baseline

The scripts in `baseline/` fine-tune the respective models on the original synthetic training sets.

### Boosting

The scripts in `boosting/` implement the boosting-inspired procedure from the paper.

After each training round:

1. false negatives are identified on the validation set,
2. their `(before, after)` error patterns are collected,
3. additional synthetic examples matching previously unseen error patterns are retrieved,
4. balanced erroneous and correct examples are added to the training set, and
5. the model is trained again on the augmented data.

The experiments use up to five boosting rounds and add up to 500 synthetic samples per newly selected error pattern.

### Random Selection

The scripts in `random_selection/` provide the random-selection ablation. Instead of selecting synthetic samples based on validation errors, the same total amount of additional synthetic data is selected randomly.

### Targeted Fine-Tuning

`targeted_finetuning/targeted_finetuning_mbert.py` performs the targeted fine-tuning experiment with mBERT.

Targeted fine-tuning with LoRA is not included because it was not part of the corresponding experiment in the paper.

### Evaluation

`evaluation/inference.py` provides a common inference script for baseline, boosting, random-selection, and targeted-fine-tuning models.

## Current Status / Planned Additions

The core training and evaluation scripts used for the main experiments are included.

The following material is still being prepared for release:

- [ ] exact split information/files needed to reproduce the original train/validation/test setup
- [ ] remaining synthetic-data pools used by the boosting and random-selection scripts
- [ ] targeted synthetic-data generation / preparation code
- [ ] scripts and prompts for the GPT-4o comparison
- [ ] environment / dependency specification
- [ ] additional documentation and complete reproduction commands

The repository is therefore currently a work in progress. Exact end-to-end reproduction of every experiment in the paper requires the additional material listed above.

## Citation

If you use this code or the accompanying dataset, please cite:

```bibtex
@article{masanti2026enhancing,
  title   = {Enhancing language models with boosting and targeted fine-tuning for real-word error detection},
  author  = {Masanti, Corina and Witschel, Hans-Friedrich and Riesen, Kaspar},
  journal = {Natural Language Processing Journal},
  volume  = {14},
  pages   = {100202},
  year    = {2026},
  doi     = {10.1016/j.nlp.2026.100202}
}
```

## License

See the `LICENSE` file for the license of the code.

The accompanying synthetic dataset is distributed separately on Hugging Face under its own license.
