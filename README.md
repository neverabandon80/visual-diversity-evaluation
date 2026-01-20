Markdown

# 📊 Visual Diversity Evaluation for Image Datasets

> **Implementation of FineVision's Visual Diversity Metric**  
> Quantifying dataset quality using SSCD embeddings, Effective Rank & Participation Ratio

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Motivation

This repository implements the **visual diversity measurement algorithm** used in HuggingFace's [FineVision](https://huggingface.co/spaces/HuggingFaceM4/FineVision) project for evaluating MLLM (Multimodal Large Language Model) SFT datasets.

### Why This Matters
- ✅ **Quantify Dataset Quality**: Transform subjective assessment into objective metrics
- ✅ **Detect Bias**: Automatically identify repetitive patterns (e.g., tracking datasets with repeated backgrounds)
- ✅ **Optimize Data Augmentation**: Understand which directions need more diversity
- ✅ **Active Learning**: Use diversity as a criterion for core-set selection

---

## 🔬 Algorithm Overview

### Step 1: SSCD Embedding Extraction
- **Model**: [SSCD (Self-Supervised Copy Detection)](https://github.com/facebookresearch/sscd-copy-detection) by Meta AI
- **Output**: 512-dimensional embedding vectors per image

### Step 2: Diversity Calculation Pipeline
Compute Covariance Matrix
→ Analyze the directional spread of data

Eigenvalue Decomposition
→ Extract principal components and their magnitudes

Calculate Effective Rank
→ Entropy-based measure of directional diversity
Effective Rank = exp(Entropy)

Calculate Participation Ratio
→ Measure how evenly variance is distributed
PR = (Σλᵢ)² / Σ(λᵢ²)

Final Diversity Score
→ Geometric Mean(Effective Rank_normalized, Participation Ratio_normalized)

text


### Reference Papers
- [The Effective Rank: A Measure of Effective Dimensionality](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf) (EUSIPCO 2007)
- [On the Importance of Single Directions for Generalization](https://arxiv.org/abs/1803.06959) (ICLR 2018)

---

## 🚀 Key Features

- ✅ **Multi-GPU Support**: Powered by `torch.nn.DataParallel`
- ✅ **Memory Efficient**: Local cache system (`.npy`) handles large-scale datasets (2.4M+ images)
- ✅ **Flexible Deployment**: CPU, Single GPU, Multi-GPU compatibility
- ✅ **Fast Processing**: Configurable batch processing with optimized throughput

---

## 📦 Installation

### Requirements
```bash
pip install -r requirements.txt
Core Dependencies
Python >= 3.8
PyTorch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
scipy >= 1.10.0
Pillow >= 9.5.0
tqdm >= 4.65.0
pyyaml >= 6.0
🎮 Quick Start
Basic Usage
Python

from embedders.sscd_embedding import SSCDEmbedder
from diversity.diversity_calculation import DiversityCalculator

# Step 1: Extract SSCD embeddings
embedder = SSCDEmbedder(device='cuda', batch_size=32)
embeddings = embedder.extract('/path/to/dataset/')

# Step 2: Calculate diversity score
calculator = DiversityCalculator()
score = calculator.calculate(embeddings)

print(f"Diversity Score: {score:.6f}")
Using Configuration Files
Bash

# CPU mode
python test.py --config configuration/config_cpu.yaml

# Single GPU
python test.py --config configuration/config_specific_gpu.yaml

# Multi-GPU
python test.py --config configuration/config_specific_multi_gpu.yaml

# Large dataset with local cache
python test.py --config configuration/config_specific_gpu_local_cache.yaml
📊 Benchmark Results
Comparison with FineVision Baselines
Dataset	Images	Diversity Score	Rating
FineVision	17.3M	0.500	⭐⭐⭐⭐⭐ Very Good
Cambrian-7M	5.4M	0.458	⭐⭐⭐⭐ Good
M4-Instruct	2.48M	0.413	⭐⭐⭐⭐ Good
Cauldron	2.0M	0.400	⭐⭐⭐⭐ Good
LLaVa-Vision	2.5M	0.298	⭐⭐⭐ Normal
Evaluation on 9 Public Datasets
Dataset	Images	Classes	Task	Diversity Score	Rating
Pascal VOC	17,125	20	Classification	0.885	⭐⭐⭐⭐⭐
V3Det	212,917	13,204	Detection	0.879	⭐⭐⭐⭐⭐
WiderFace	16,106	1	Face Detection	0.813	⭐⭐⭐⭐⭐
CrowdHuman	23,740	2	Human Detection	0.758	⭐⭐⭐⭐⭐
M4-Instruct	2,481,646	Multi	MLLM Instruction	0.413	⭐⭐⭐⭐
RVSD	8,404	80 scenes	DeSnowing	0.293	⭐⭐
SeaDroneSee	14,227	6	Maritime Detection	0.183	⭐
DanceTrack	38,551	1	Human Tracking	0.145	⭐
R7_Tracking	6,000	1	Sports Tracking	0.071	⭐
Score Interpretation Guide
text

Diversity Score >= 0.50        : ⭐⭐⭐⭐⭐ Very Good
                                 - FineVision benchmark level
                                 - Optimal for large-scale MLLM training

0.40 <= Score < 0.50           : ⭐⭐⭐⭐ Good
                                 - Cambrian-7M level
                                 - Suitable for general MLLM training

0.30 <= Score < 0.40           : ⭐⭐⭐ Normal
                                 - LLaVa-Vision level
                                 - Consider filtering or augmentation

0.20 <= Score < 0.30           : ⭐⭐ Low
                                 - Potential bias or duplication issues

Score < 0.20                   : ⭐ Very Low
                                 - Quality inspection required
💡 Key Insights
1. Object Detection Datasets
Most achieve Very Good diversity (score > 0.7)
V3Det: 13,204 classes → Maintains 0.879 diversity despite 213K images
2. Tracking Datasets
Repetitive backgrounds lead to Very Low diversity (0.071 ~ 0.183)
R7_Tracking: 3 backgrounds × 2000 frames → Severe bias (0.071)
3. MLLM Datasets
M4-Instruct: Lower than FineVision (0.5) but still Good (0.413)
Category diversity significantly impacts the score
4. Interesting Cases
RVSD: 80 locations → Higher diversity (0.293) despite being a tracking dataset
SeaDroneSee: Repetitive maritime background → Lower diversity (0.183) despite detection task
🗂 Project Structure
text

visual-diversity-evaluation/
├── configuration/              # YAML configuration files
│   ├── config_cpu.yaml
│   ├── config_specific_gpu.yaml
│   ├── config_specific_multi_gpu.yaml
│   └── config_specific_gpu_local_cache.yaml
│
├── data_loaders/
│   └── custom_dataset.py      # Custom image dataset loader
│
├── embedders/
│   └── sscd_embedding.py      # SSCD embedding extractor
│
├── diversity/
│   └── diversity_calculation.py  # Diversity metric implementation
│
├── utils.py                    # Utility functions
├── test.py                     # Main evaluation script
├── requirements.txt
└── README.md
🎯 Use Cases
1. Dataset Quality Assessment
Python

# Evaluate diversity of your custom dataset
score = evaluate_diversity('/path/to/my_dataset/')
print(f"Dataset Quality Score: {score:.3f}")
2. Data Augmentation Guidance
Python

# Analyze which directions need augmentation
effective_rank, participation_ratio = get_diversity_components(embeddings)
3. Active Learning Core-Set Selection
Python

# Select diverse samples for annotation
selected_indices = select_diverse_samples(embeddings, k=1000)
4. Multi-Dataset Comparison
Python

# Compare multiple datasets
scores = {
    'dataset_A': evaluate_diversity('/path/A/'),
    'dataset_B': evaluate_diversity('/path/B/'),
}
📈 Performance Benchmarks
Environment	Batch Size	Throughput (imgs/sec)	Memory
CPU (16 cores)	4	~5	8GB RAM
RTX 2080 (Single)	32	~120	8GB VRAM
RTX 2080 (×4)	128	~400	32GB VRAM
A100 (Single)	64	~300	40GB VRAM
Large-Scale Dataset Handling:

M4-Instruct (2.48M images) → Uses local cache (.npy format)
Batch size 4, Single RTX 2080 → ~8 hours processing time
🔧 Advanced Configuration
Multi-GPU Setup
YAML

# configuration/config_specific_multi_gpu.yaml
device: 'cuda'
gpu_ids: [0, 1, 2, 3]
batch_size: 128
use_data_parallel: true
num_workers: 8
Local Cache for Large Datasets
YAML

# configuration/config_specific_gpu_local_cache.yaml
use_local_cache: true
cache_dir: './results/embeddings/'
cache_format: 'npy'
overwrite_cache: false
🤝 Contributing
Contributions are welcome! Please feel free to:

Submit bug reports or feature requests via Issues
Create Pull Requests for improvements
Share your evaluation results on new datasets
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
This work builds upon:

FineVision - HuggingFace M4 Team
SSCD - Meta AI Research
Effective Rank - Roy & Vetterli (EUSIPCO 2007)
Participation Ratio - Morcos et al. (ICLR 2018)
