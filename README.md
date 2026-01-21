````markdown
# 📊 Visual Diversity Evaluation for Image Datasets

> **FineVision Visual Diversity Metric Implementation**  
> Quantifying image dataset quality using SSCD embeddings, Effective Rank, and Participation Ratio

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository provides a reproducible implementation of the **visual diversity evaluation metric**
used in HuggingFace **FineVision** for assessing **MLLM (Multimodal Large Language Model) SFT datasets**.

The method converts qualitative visual diversity into a **single quantitative score**
by analyzing the geometric structure of SSCD embedding distributions.

---

## 🎯 Motivation

### Why Visual Diversity Matters

- Objective dataset quality measurement
- Automatic bias and duplication detection
- Guidance for data augmentation strategies
- Core-set selection for active learning
- Fair comparison between large-scale datasets

---

## 🔬 Methodology

### 1. SSCD Embedding Extraction

- **Model**: SSCD (Self-Supervised Copy Detection, Meta AI)
- **Embedding Dimension**: 512
- **Property**: Robust to near-duplicate and semantic similarity

---

### 2. Diversity Computation Pipeline

#### Step 1 — Covariance Estimation
Computes second-order statistics of the embedding distribution to capture overall variance structure.

#### Step 2 — Eigenvalue Decomposition
Extracts principal directions and corresponding variance magnitudes from the covariance matrix.

#### Step 3 — Effective Rank (ER)
Measures the **effective dimensionality** of the embedding space using an entropy-based criterion.
Higher values indicate that variance is spread across more independent directions.

#### Step 4 — Participation Ratio (PR)
Measures how **evenly variance is distributed** across embedding dimensions.
Higher values indicate balanced usage of dimensions rather than dominance by a few directions.

#### Step 5 — Final Diversity Score
Combines normalized Effective Rank and Participation Ratio into a **single diversity score**
that reflects both dimensional richness and variance balance.

---

## 📚 References

- Roy & Vetterli, *The Effective Rank*, EUSIPCO 2007
- Morcos et al., *On the Importance of Single Directions for Generalization*, ICLR 2018
- Meta AI, *SSCD: Self-Supervised Copy Detection*

---

## ✨ Key Features

- Multi-GPU inference via `torch.nn.DataParallel`
- Scales to millions of images
- Local embedding cache (`.npy`) for memory efficiency
- CPU / Single GPU / Multi-GPU compatible
- Reproducible and deterministic evaluation

---

## 📦 Installation

### Install Dependencies

```bash
pip install -r requirements.txt
````

### Core Dependencies

* **Python** ≥ 3.8
* **PyTorch** ≥ 2.0.0
* **torchvision** ≥ 0.15.0
* **numpy** ≥ 1.24.0
* **scipy** ≥ 1.10.0
* **Pillow** ≥ 9.5.0
* **tqdm** ≥ 4.65.0
* **pyyaml** ≥ 6.0

---

## 🚀 Quick Start

### Minimal Example

```python
from embedders.sscd_embedding import SSCDEmbedder
from diversity.diversity_calculation import DiversityCalculator

embedder = SSCDEmbedder(device="cuda", batch_size=32)
embeddings = embedder.extract("/path/to/images")

calculator = DiversityCalculator()
score = calculator.calculate(embeddings)

print(f"Diversity Score: {score:.4f}")
```

---

## ⚙️ Configuration-Based Execution

```bash
# CPU
python test.py --config configuration/config_cpu.yaml

# Single GPU
python test.py --config configuration/config_specific_gpu.yaml

# Multi-GPU
python test.py --config configuration/config_specific_multi_gpu.yaml

# Large-scale dataset with local cache
python test.py --config configuration/config_specific_gpu_local_cache.yaml
```

---

## 📊 Benchmark Results

### FineVision-Scale Dataset Comparison

| Dataset      | Images | Diversity Score | Rating |
| ------------ | ------ | --------------- | ------ |
| FineVision   | 17.3M  | 0.500           | ⭐⭐⭐⭐⭐  |
| Cambrian-7M  | 5.4M   | 0.458           | ⭐⭐⭐⭐   |
| M4-Instruct  | 2.48M  | 0.413           | ⭐⭐⭐⭐   |
| Cauldron     | 2.0M   | 0.400           | ⭐⭐⭐⭐   |
| LLaVA-Vision | 2.5M   | 0.298           | ⭐⭐⭐    |

---

## 📈 Public Dataset Evaluation

| Dataset     | Task           | Diversity | Interpretation |
| ----------- | -------------- | --------- | -------------- |
| Pascal VOC  | Classification | 0.885     | Very High      |
| V3Det       | Detection      | 0.879     | Very High      |
| WiderFace   | Face Detection | 0.813     | Very High      |
| CrowdHuman  | Detection      | 0.758     | Very High      |
| RVSD        | DeSnowing      | 0.293     | Low            |
| SeaDroneSee | Detection      | 0.183     | Very Low       |
| DanceTrack  | Tracking       | 0.145     | Very Low       |
| R7_Tracking | Tracking       | 0.071     | Extremely Low  |

---

## 🧭 Diversity Score Interpretation

| Score Range | Meaning                    |
| ----------- | -------------------------- |
| ≥ 0.50      | FineVision-level diversity |
| 0.40 – 0.50 | Suitable for MLLM training |
| 0.30 – 0.40 | Augmentation recommended   |
| 0.20 – 0.30 | Strong bias suspected      |
| < 0.20      | Severe redundancy          |

---

## 🗂 Project Structure

```text
visual-diversity-evaluation/
├── configuration/
│   ├── config_cpu.yaml
│   ├── config_specific_gpu.yaml
│   ├── config_specific_multi_gpu.yaml
│   └── config_specific_gpu_local_cache.yaml
├── data_loaders/
│   └── custom_dataset.py
├── embedders/
│   └── sscd_embedding.py
├── diversity/
│   └── diversity_calculation.py
├── utils.py
├── test.py
├── requirements.txt
└── README.md
```

---

## 🎯 Use Cases

### Dataset Quality Assessment

```python
score = evaluate_diversity("/path/to/dataset")
```

### Augmentation Direction Analysis

```python
effective_rank, participation_ratio = get_diversity_components(embeddings)
```

### Active Learning Core-set Selection

```python
selected_indices = select_diverse_samples(embeddings, k=1000)
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

* HuggingFace M4 – FineVision
* Meta AI – SSCD
* Roy & Vetterli – Effective Rank
* Morcos et al. – Participation Ratio

```

---

```
