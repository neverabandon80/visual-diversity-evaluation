
````markdown
# 📊 Visual Diversity Evaluation for Image Datasets

> **FineVision Visual Diversity Metric Implementation**  
> Quantifying image dataset quality using **SSCD embeddings**, **Effective Rank**, and **Participation Ratio**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository provides a **reproducible implementation** of the **visual diversity evaluation metric**
used in HuggingFace **FineVision** for assessing **MLLM (Multimodal Large Language Model) SFT datasets**.

The metric converts *qualitative visual diversity* into a **quantitative scalar score** using
representation geometry analysis on **SSCD embeddings**.

---

## 🎯 Motivation

### Why Visual Diversity Matters

- **Objective dataset quality measurement**
- **Automatic bias & duplication detection**
- **Guidance for data augmentation**
- **Core-set selection for active learning**
- **Fair comparison between large-scale datasets**

---

## 🔬 Methodology

### 1️⃣ SSCD Embedding Extraction

- **Model**: SSCD (Self-Supervised Copy Detection, Meta AI)
- **Embedding Dimension**: 512
- **Property**: Robust to near-duplicate and semantic similarity

---

### 2️⃣ Diversity Computation Pipeline

#### Step 1 — Covariance Estimation
- Computes second-order statistics of embedding distribution

#### Step 2 — Eigenvalue Decomposition
- Extracts principal directions and variance magnitudes

#### Step 3 — Effective Rank (ER)
- Measures **entropy-based intrinsic dimensionality**

#### Step 4 — Participation Ratio (PR)

* Measures **variance balance across dimensions**

#### Step 5 — Final Diversity Score

* Geometric mean of normalized ER and PR

---

## 📚 References

* Roy & Vetterli, *The Effective Rank*, EUSIPCO 2007
* Morcos et al., *On the Importance of Single Directions*, ICLR 2018
* Meta AI, *SSCD: Self-Supervised Copy Detection*

---

## ✨ Key Features

* **Multi-GPU inference** via `torch.nn.DataParallel`
* **Scales to millions of images**
* **Local embedding cache** (`.npy`) for memory efficiency
* **CPU / Single GPU / Multi-GPU compatible**
* **Reproducible & deterministic evaluation**

---

## 📦 Installation

### Requirements

```bash
pip install -r requirements.txt
```

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

# Large-scale dataset (cached)
python test.py --config configuration/config_specific_gpu_local_cache.yaml
```

---

## 📊 Benchmark Results

### FineVision-Scale Dataset Comparison

| Dataset      | Images | Score | Rating |
| ------------ | ------ | ----- | ------ |
| FineVision   | 17.3M  | 0.500 | ⭐⭐⭐⭐⭐  |
| Cambrian-7M  | 5.4M   | 0.458 | ⭐⭐⭐⭐   |
| M4-Instruct  | 2.48M  | 0.413 | ⭐⭐⭐⭐   |
| Cauldron     | 2.0M   | 0.400 | ⭐⭐⭐⭐   |
| LLaVA-Vision | 2.5M   | 0.298 | ⭐⭐⭐    |

---

## 📈 Public Dataset Evaluation

| Dataset     | Task           | Score | Interpretation |
| ----------- | -------------- | ----- | -------------- |
| Pascal VOC  | Classification | 0.885 | Very High      |
| V3Det       | Detection      | 0.879 | Very High      |
| WiderFace   | Face Detection | 0.813 | Very High      |
| CrowdHuman  | Detection      | 0.758 | Very High      |
| DanceTrack  | Tracking       | 0.145 | Very Low       |
| R7 Tracking | Tracking       | 0.071 | Extremely Low  |

---

## 🧭 Score Interpretation Guide

| Score Range | Meaning                    |
| ----------- | -------------------------- |
| ≥ 0.50      | FineVision-level diversity |
| 0.40 – 0.50 | Suitable for MLLM training |
| 0.30 – 0.40 | Requires augmentation      |
| 0.20 – 0.30 | Strong bias suspected      |
| < 0.20      | Severe redundancy          |

---

## 🗂 Project Structure

```text
visual-diversity-evaluation/
├── configuration/
├── data_loaders/
├── embedders/
├── diversity/
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
er, pr = get_diversity_components(embeddings)
```

### Active Learning (Core-set Selection)

```python
indices = select_diverse_samples(embeddings, k=1000)
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

원하는 방향 말해줘.
```
