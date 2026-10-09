
# NeuroTumourAI
### An Adaptive Hybrid Deep Learning Framework with Explainability for Precise MRI-Based Brain Tumour Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red)
![Medical AI](https://img.shields.io/badge/Domain-Medical%20Imaging-purple)
![Explainable AI](https://img.shields.io/badge/AI-Explainable-success)
![Research](https://img.shields.io/badge/Project-Research-orange)

**NeuroTumourAI** is a research-oriented hybrid deep learning framework for MRI-based classification of brain gliomas into Low-Grade Glioma (LGG) and High-Grade Glioma (HGG). It combines three-dimensional convolutional neural networks, radiomic descriptors, morphometric characteristics, and graph-based spatial representations using gated cross-modal attention and Mixture-of-Experts fusion.

The framework integrates tumour-aware segmentation, explainable AI, probability calibration, and subject-level evaluation to support interpretable and reproducible medical imaging research.

> **Research purpose:** This project is intended for research and experimentation. It is not independently validated or approved for clinical diagnosis or treatment decisions.

---

## Overview

Brain gliomas are brain tumours that require careful analysis of medical imaging data. Conventional image-based classification approaches may not fully exploit complementary information from tumour appearance, shape, texture, and spatial structure.

NeuroTumourAI investigates a multimodal learning strategy that combines deep MRI features with handcrafted radiomic and morphometric descriptors and graph-based representations.

The proposed core architecture, **HybridTumourNet**, uses a Gated Cross-Modal Attention Transformer (GCAT) and a Gated Mixture-of-Experts (G-MoE) mechanism to integrate complementary feature representations for LGG/HGG classification.

The framework also incorporates auxiliary tumour segmentation, model interpretation, and probability calibration to support systematic evaluation.

## Key Features

- **Multimodal MRI processing:** Supports T1, T1CE, T2, and FLAIR MRI modalities.
- **Tumour-aware segmentation:** Auxiliary 3D U-Net for Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET) regions.
- **3D deep feature extraction:** CNN-based representations with Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM).
- **Radiomic feature analysis:** Intensity, texture, and imaging descriptors.
- **Morphometric analysis:** Shape and geometric characteristics.
- **Graph-based representation learning:** Supervoxel graphs and GraphSAGE embeddings.
- **Cross-modal feature fusion:** Gated Cross-Modal Attention Transformer.
- **Mixture-of-Experts classification:** Gated specialist networks for adaptive feature integration.
- **Explainable AI:** Grad-CAM++, SHAP, and graph saliency utilities.
- **Probability calibration:** Expected Calibration Error (ECE) and temperature scaling.
- **Subject-level evaluation:** Patient-aware data partitioning and subject-level prediction aggregation.
- **External evaluation design:** Supports evaluation workflows involving independent TCGA cohorts.

---

## Research Objectives

The framework is designed around the following research goals:

1. Integrate complementary deep, radiomic, morphometric, and graph-based MRI representations.
2. Investigate adaptive cross-modal attention and expert-based feature fusion for glioma classification.
3. Incorporate tumour-aware segmentation into the MRI analysis pipeline.
4. Provide interpretable predictions through spatial and feature-level explanation methods.
5. Support calibrated prediction probabilities and patient-level evaluation.
6. Establish a reproducible evaluation workflow with appropriate internal validation and independent external testing.

These are research goals; their achievement must be established through experiments and verified implementation results.

---

## Proposed Architecture

The proposed NeuroTumourAI workflow consists of the following stages:

```text
Multimodal Brain MRI
  |
  +-- T1
  +-- T1CE
  +-- T2
  +-- FLAIR
  |
  v
MRI Preprocessing
  |
  +-- N4 Bias Correction
  +-- Brain Masking
  +-- Image Registration
  +-- Intensity Normalization
  +-- Tumour-Aware Patch Generation
  |
  v
Auxiliary Tumour Segmentation
  |
  +-- 3D U-Net
  +-- Whole Tumour (WT)
  +-- Tumour Core (TC)
  +-- Enhancing Tumour (ET)
  |
  v
Multimodal Feature Extraction
  |
  +-- 3D CNN Features
  +-- Radiomic Features
  +-- Morphometric Features
  +-- GraphSAGE Features
  |
  v
Feature Tokenization
  |
  v
Gated Cross-Modal Attention Transformer
(GCAT)
  |
  v
Gated Mixture-of-Experts
(G-MoE)
  |
  v
LGG / HGG Classification
  |
  +-- Prediction Probabilities
  +-- Probability Calibration
  +-- Subject-Level Evaluation
  +-- Grad-CAM++ Explanations
  +-- SHAP Feature Attributions
  +-- Graph Saliency
```

### Core Architecture Components

| Component | Role |
|---|---|
| 3D CNN | Learns spatial and volumetric MRI representations |
| SE and CBAM | Refine feature-channel and spatial attention |
| Radiomics | Represents intensity and texture characteristics |
| Morphometrics | Represents tumour shape and geometry |
| GraphSAGE | Learns graph-based spatial representations |
| GCAT | Models relationships among modality-specific feature tokens |
| G-MoE | Combines specialized expert representations through gating |
| Auxiliary 3D U-Net | Supports tumour-region segmentation |
| Calibration module | Adjusts prediction probabilities using temperature scaling |
| Explainability module | Provides spatial, feature-level, and graph-based attributions |

The reference architecture specifies four independently projected 256-dimensional feature tokens, three GCAT attention layers, eight attention heads, and four gated experts.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python 3.10+ |
| Deep learning | PyTorch |
| Medical image processing | SimpleITK, NiBabel |
| Radiomics | PyRadiomics |
| Graph neural networks | PyTorch Geometric |
| Image processing | scikit-image, SciPy |
| Machine learning | scikit-learn |
| Explainable AI | SHAP, custom attribution utilities |
| Visualization | Matplotlib |
| Statistical analysis | SciPy |
| Testing | pytest |
| Recommended hardware | CUDA-enabled GPU |

---

## Datasets

The research workflow identifies the following datasets.

| Dataset | Intended purpose |
|---|---|
| BraTS 2024 | Model development and internal validation, subject to cohort eligibility |
| TCGA-GBM | External evaluation |
| TCGA-LGG | External evaluation |

### Dataset Sources

- **BraTS:** https://www.synapse.org/
- **The Cancer Imaging Archive (TCIA):** https://www.cancerimagingarchive.net/
- **TCGA-GBM:** https://www.cancerimagingarchive.net/collection/tcga-gbm/
- **TCGA-LGG:** https://www.cancerimagingarchive.net/collection/tcga-lgg/

### Dataset Preparation Requirements

Prepare a CSV manifest containing subject identifiers, diagnostic labels, and MRI file paths.

The expected columns are:

```csv
subject_id,label,t1,t1ce,t2,flair,seg
```

Where:

- `subject_id`: Unique subject identifier.
- `label`: Verified classification label, `LGG` or `HGG`.
- `t1`: T1 MRI file path.
- `t1ce`: Contrast-enhanced T1 MRI file path.
- `t2`: T2 MRI file path.
- `flair`: FLAIR MRI file path.
- `seg`: Reference segmentation file path, when available.

Use valid diagnostic labels and consistent subject identifiers. Segmentation labels alone do not establish glioma grade. Confirm cohort eligibility and diagnostic-label provenance before training.

**Dataset access:** MRI datasets are not included in this repository. Users must obtain them from their respective providers and comply with the applicable access requirements and usage agreements.

---

## Project Structure

The following is a suggested logical structure for the framework. Adapt it to the actual files included in the repository.

```text
NeuroTumourAI/
|
+-- data/
|   +-- development.csv
|   +-- external.csv
|
+-- scripts/
|   +-- neurotumourai_cli.py
|   +-- train_segmentation.py
|   +-- normalize_features.py
|
+-- outputs/
|   +-- folds/
|   +-- features/
|   +-- models/
|   +-- evaluation/
|
+-- tests/
|
+-- requirements.txt
+-- pyproject.toml
+-- README.md
```

Dataset files, trained models, intermediate features, and generated outputs should be stored separately from the source code where practical. The structure above is illustrative and should not be interpreted as a verified inventory of every file in the repository.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/cherukuMurali-k/NeuroTumourAI
cd NeuroTumourAI
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install a PyTorch build compatible with your Python version and CUDA environment, if applicable.

```bash
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Run these installation commands only when the corresponding project configuration files are present. Resolve any dependency conflicts according to the installed Python and PyTorch versions.

### 4. Verify the Environment

If the relevant test and CLI files are available, run:

```bash
pytest -q
python scripts/neurotumourai_cli.py smoke
```

Successful installation does not, by itself, establish that model training or the complete research workflow is functioning correctly.

---

## Dataset Preparation

### Step 1: Prepare the Manifest

Create development and external-evaluation CSV files containing the required MRI paths and verified diagnostic labels.

### Step 2: Generate Patient-Level Folds

```bash
python scripts/neurotumourai_cli.py folds \
  --manifest data/development.csv \
  --external data/external.csv \
  --output outputs/folds
```

Keep all patches and derived samples from a single subject within the same data partition to prevent patient-level leakage.

### Step 3: Prepare Training Features

```bash
python scripts/neurotumourai_cli.py prepare \
  --manifest outputs/folds/fold1_train.csv \
  --output outputs/features_train1 \
  --fit-nyul \
  --allow-training-masks
```

### Step 4: Prepare Validation Features

```bash
python scripts/neurotumourai_cli.py prepare \
  --manifest outputs/folds/fold1_val.csv \
  --output outputs/features_val1 \
  --nyul outputs/features_train1/nyul.npz \
  --segmenter outputs/seg_fold1/segmenter.pt \
  --feature-schema outputs/features_train1/radiomics_features.json
```

Use preprocessing parameters fitted on the training partition. Held-out subjects should use predicted tumour masks during feature extraction rather than reference segmentation masks, unless a different evaluation setting is explicitly defined.

The commands above represent the documented workflow and require the relevant scripts, input files, and configuration artifacts to be present.

---

## Execution Workflow

The following steps describe the intended training and evaluation sequence.

### Step 1: Train the Auxiliary Segmentation Model

```bash
python scripts/train_segmentation.py \
  --train outputs/folds/fold1_train.csv \
  --val outputs/folds/fold1_val.csv \
  --output outputs/seg_fold1 \
  --epochs 100
```

### Step 2: Fit Feature Normalization

```bash
python scripts/normalize_features.py fit \
  --features outputs/features_train1/features.csv \
  --output outputs/fold1_scalers.npz
```

### Step 3: Transform Training Features

```bash
python scripts/normalize_features.py transform \
  --features outputs/features_train1/features.csv \
  --scalers outputs/fold1_scalers.npz \
  --output outputs/features_train1_scaled
```

### Step 4: Transform Validation Features

```bash
python scripts/normalize_features.py transform \
  --features outputs/features_val1/features.csv \
  --scalers outputs/fold1_scalers.npz \
  --output outputs/features_val1_scaled
```

### Step 5: Train HybridTumourNet

```bash
python scripts/neurotumourai_cli.py train \
  --train outputs/features_train1_scaled/features.csv \
  --val outputs/features_val1_scaled/features.csv \
  --output outputs/model_fold1 \
  --epochs 200
```

### Step 6: Calibrate Prediction Probabilities

```bash
python scripts/neurotumourai_cli.py calibrate \
  --model outputs/model_fold1/best.pt \
  --features outputs/features_val1_scaled/features.csv \
  --output outputs/model_fold1/temperature.json
```

Fit calibration parameters on a dedicated calibration partition or use cross-fitting when appropriate. Avoid using the same subjects for both fitting and unbiased assessment of calibration performance.

### Step 7: Evaluate the Model

```bash
python scripts/neurotumourai_cli.py evaluate \
  --model outputs/model_fold1/best.pt \
  --features outputs/features_val1_scaled/features.csv \
  --output outputs/model_fold1/eval \
  --temperature outputs/model_fold1/temperature.json
```

**Important:** Verify each command against the actual scripts and CLI arguments in the repository before running the complete workflow. File names, paths, and outputs may require adjustment to match the implementation.

---

## Training Configuration

The following values are reference settings documented for the proposed architecture. They should be checked against the executable configuration before claiming exact reproduction.

| Parameter | Reference value |
|---|---|
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Maximum epochs | 200 |
| Learning-rate schedule | 10-epoch warm-up followed by cosine annealing |
| Effective batch size | 4 |
| Early-stopping patience | 20 |
| Embedding dimension | 256 |
| Attention heads | 8 |
| GCAT layers | 3 |
| G-MoE experts | 4 |
| Classification classes | 2 |
| Random seed | 42 |

Actual training outcomes depend on dataset composition, preprocessing, hardware, random seeds, and implementation details. No specific accuracy or clinical performance is guaranteed.

---

## Evaluation Metrics

### Classification Metrics

The evaluation framework identifies the following metrics:

- Accuracy
- Precision
- Recall
- Macro-F1 score
- Sensitivity
- Specificity
- Area Under the Receiver Operating Characteristic Curve (AUROC)
- Confusion matrix

### Segmentation Metrics

For WT, TC, and ET regions:

- Dice Similarity Coefficient (Dice)
- Intersection over Union (IoU)

### Calibration Metrics

- Expected Calibration Error (ECE)
- Temperature scaling

### Subject-Level Evaluation

Aggregate patch-level predictions at the subject level and evaluate each subject once. Ensure that the evaluation protocol prevents overlap between training, validation, calibration, and external-test subjects.

Report measured results only after running the experiments and validating the outputs.

---

## Explainable AI

NeuroTumourAI includes complementary approaches for interpreting model predictions.

### 1. Grad-CAM++

Provides spatial attribution maps that help investigate which MRI regions contribute to a model's prediction.

### 2. SHAP

Estimates the contributions of radiomic and morphometric features to model outputs.

### 3. Graph Saliency

Investigates the contribution of tumour supervoxels and graph structures to graph-based representations and predictions.

### 4. Probability Calibration

Temperature scaling is used to adjust prediction confidence. Calibration should be assessed on appropriately separated data.

**Interpretation note:** Explanations are exploratory tools. Attribution maps and feature importance scores do not independently establish biological relevance, causal relationships, or clinical reliability.

---

## Reproducibility and Validation

Before claiming complete reproduction of the proposed framework, verify:

- [ ] Diagnostic-label provenance and eligible BraTS classification cohorts.
- [ ] MRI registration, intensity normalization, and brain-masking quality.
- [ ] Leakage-free patient-level cross-validation across all intended folds.
- [ ] Radiomic extraction, wavelet features, and feature-selection implementation.
- [ ] Segmentation quality and use of predicted masks for held-out subjects.
- [ ] Model baselines and ablation experiments.
- [ ] Independent external evaluation on eligible TCGA cohorts.
- [ ] Explanation fidelity and statistical reporting.
- [ ] Calibration using an appropriate independent or cross-fitted procedure.
- [ ] Reproducibility of reported results from the executable configuration.

The documented strategy trains auxiliary segmentation separately from the classifier. It should not be described as simultaneous end-to-end optimization of segmentation and classification unless that behavior is implemented and experimentally verified.

---

## Research Applications

Potential research applications include:

- MRI-based glioma classification.
- Multimodal medical image analysis.
- Hybrid radiomics and deep learning.
- Graph-based tumour representation learning.
- Cross-modal attention and expert-based fusion.
- Explainable artificial intelligence for medical imaging.
- Prediction probability calibration.
- Retrospective cross-dataset evaluation.

These applications represent research directions rather than claims of validated clinical utility.

---

## Limitations

- The framework requires suitable MRI data and reliable diagnostic labels.
- Full-resolution 3D training may require substantial GPU memory and computation.
- MRI acquisition differences can affect cross-dataset generalization.
- Segmentation errors can influence downstream feature extraction.
- Radiomic and morphometric features depend on preprocessing and extraction settings.
- Explainability methods may produce incomplete or unstable attributions.
- External validation and reproducibility must be established experimentally.
- Clinical performance and clinical utility have not been independently established by this documentation.

---

## Data Availability

The MRI datasets are not included in this repository.

Users must obtain datasets from the relevant providers and comply with their licensing, access, privacy, and data-use requirements. Do not upload restricted medical data, personally identifiable information, or patient-level records without appropriate authorization.

An explicit software license should be added before publicly redistributing the code under an open-source license.

---

## Citation

**Manuscript title:**

*An Adaptive Hybrid Deep Learning Framework with Explainability for Precise MRI-Based Brain Tumour Classification*

**Framework:** NeuroTumourAI

**Core architecture:** HybridTumourNet

---

## License

This project is licensed under the MIT License. You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the terms and conditions of the MIT License.

---

## Acknowledgement

NeuroTumourAI explores the integration of multimodal MRI analysis, hybrid deep learning, graph-based representation learning, and explainable AI for glioma classification research.

**NeuroTumourAI — Towards Interpretable and Reproducible MRI-Based Brain Tumour Classification.**
