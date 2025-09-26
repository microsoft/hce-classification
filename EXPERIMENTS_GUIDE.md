# Experiments Guide

This guide provides detailed documentation of the codebase structure and step-by-step instructions for reproducing the experiments from the paper "Hierarchical cross-entropy loss improves atlas-scale single-cell annotation models".

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Hierarchical Cross-Entropy Implementation](#hierarchical-cross-entropy-implementation)
3. [Data Requirements](#data-requirements)
4. [Reproducing Experiments](#reproducing-experiments)
5. [Model Evaluation](#model-evaluation)
6. [Understanding the Results](#understanding-the-results)

## Repository Structure

This repository is organized in the following way:

```
hce-classification/
├── scTab/                          # Source code
├── model_training/                 # Training scripts and configurations
│   ├── train_linear.py             # Linear classifier training
│   ├── train_mlp.py                # MLP classifier training
│   ├── train_tabnet.py             # TabNet classifier training
│   ├── train_*_hier_seed0.sh       # Shell scripts for HCE training
│   └── train_utils.py              # Training utilities
├── model_evaluation/               # Evaluation scripts
│   ├── model_evaluation.py         # Evaluation script for OOD test set (study-split)
│   ├── model_evaluation_sctab.py   # Evaluation script for ID test set (patient-split)
│   ├── checkpoint_list.txt         # Trained model checkpoints
│   └── utils.py                    # Evaluation utilities
└── example_ce_vs_hce.ipynb         # Standalone HCE demonstration
```

## Hierarchical Cross-Entropy Implementation

This repository is based on the [scTab study](https://github.com/theislab/scTab) but has been modified to implement hierarchical cross-entropy loss.

### Implementation Details:
- The `BaseClassifier` class in `scTab/cellnet/models.py` has been extended to include the hierarchical loss function:
```python
    def _hierarchical_loss(self, logits, targets, weight=None):
        cell_type_probs = torch.softmax(logits, dim=-1)
        cell_type_probs = torch.matmul(cell_type_probs, self.child_lookup_transposed)
        cell_type_probs = torch.log(
            cell_type_probs + torch.tensor(1e-6, device=cell_type_probs.device)
        )
        res = F.nll_loss(cell_type_probs, targets, weight=weight)
        return res
```

- All classifier models (Linear, MLP, TabNet) inherit this functionality and accept a `use_hierarchical_loss` flag that switches between standard cross-entropy and hierarchical cross-entropy loss during training
- The `child_lookup_transposed` matrix encodes the hierarchical relationships between cell types (equivalent to the transposedreachability matrix from the README example)

## Data Requirements

### Training Data (Required)
The models require the CELLxGENE census version "2023-05-15" preprocessed by scTab:

```bash
# Download training data (manually required)
wget https://pklab.med.harvard.edu/felix/data/merlin_cxg_2023_05_15_sf-log1p.tar.gz
tar -xzf merlin_cxg_2023_05_15_sf-log1p.tar.gz
```


### Evaluation Data (Automatic)
Evaluation uses CELLxGENE census "2023-12-15", automatically downloaded via the CELLxGENE API.

## Reproducing Experiments

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set data path (adjust to your download location)
export DATA_PATH="/path/to/merlin_cxg_2023_05_15_sf-log1p"
```

### 2. Training Models

The repository provides training scripts for each model:

#### Linear Classifier with HCE:
```bash
cd model_training
bash train_linear_hier_seed0.sh
```

#### MLP Classifier with HCE:
```bash
cd model_training  
bash train_mlp_hier_seed0.sh
```

#### TabNet Classifier with HCE:
```bash
cd model_training
bash train_tabnet_hier_seed0.sh
```

#### Training Baseline (CE) Models:
To train standard cross-entropy baselines, modify the shell scripts by changing:
```bash
--use_hierarchical_loss True  # Change to False
```

#### Multiple Runs:
To run multiple seeds, duplicate the shell scripts and change the `--seed` argument accordingly.

### 3. Understanding Training Output

Training creates checkpoints and logs in the following structure:
```
$DATA_PATH/tb_logs/
└── {model_name}_hierarchical_loss/    # HCE models include suffix
    └── default/version_0/
        ├── checkpoints/               # Best model checkpoints
        ├── hparams.yaml              # Hyperparameters
        └── events.out.tfevents.*     # TensorBoard logs
```

## Model Evaluation

> The trained model checkpoints and evaluation results are available on Zenodo: [https://zenodo.org/records/XXXXXX](https://zenodo.org/records/XXXXXX)

### 1. Checkpoint Preparation

Update `model_evaluation/checkpoint_list.txt` with paths to your trained models:
```
/path/to/cxg_2023_05_15_linear_hierarchical_loss/checkpoints/best.ckpt
/path/to/cxg_2023_05_15_mlp_hierarchical_loss/checkpoints/best.ckpt  
/path/to/cxg_2023_05_15_tabnet_hierarchical_loss/checkpoints/best.ckpt
```

### 2. Running OOD Evaluation

#### Bulk Evaluation (Recommended):
```bash
cd model_evaluation
bash model_evaluation_bulk.sh checkpoint_list.txt
```

### 3. Running ID Evaluation

To evaluate in-distribution (ID) performance on patient-split data:
```bash
cd model_evaluation
bash model_evaluation_sctab_bulk.sh checkpoint_list.txt
```

## Understanding the Results

The evaluation framework produces comprehensive results with two main evaluation types:

### 1. Out-of-Distribution (OOD) Evaluation
**Script**: `model_evaluation.py` via `model_evaluation_bulk.sh`  
**Purpose**: Evaluates models on new datasets not seen during training (study-split evaluation)  
**Data**: CELLxGENE census "2023-12-15" datasets not in training set

### 2. In-Distribution (ID) Evaluation  
**Script**: `model_evaluation_sctab.py` via `model_evaluation_sctab_bulk.sh`  
**Purpose**: Evaluates models on held-out patients from training datasets (patient-split evaluation)  
**Data**: Test split from CELLxGENE census "2023-05-15" training data

### Results Directory Structure

Training and evaluation results are organized in the following hierarchy:

```
$DATA_PATH/tb_logs/
├── {model_name}/                          # Standard CE models (no suffix)
│   └── default/
│       └── version_{0,1,2,3}/             # Multiple random seeds
│           ├── checkpoints/
│           │   └── val_f1_macro_epoch=X_val_f1_macro=Y.{Z}/
│           │       ├── {model}_detailed_results.parquet
│           │       ├── {model}_overall_metrics.csv
│           │       ├── {model}_per_assay_metrics.csv
│           │       ├── {model}_per_class_metrics.csv
│           │       ├── {model}_per_dataset_metrics.csv
│           │       ├── {model}_per_disease_metrics.csv
│           │       ├── {model}_per_tissue_general_metrics.csv
│           │       ├── {model}_per_tissue_metrics.csv
│           ├── sctab_test/                 # ID evaluation results
│           │   └── val_f1_macro_epoch=X_val_f1_macro=Y.{Z}/
│           │       ├── {model}_overall_metrics.csv
│           │       ├── {model}_per_class_metrics.csv
│           ├── hparams.yaml
│           └── events.out.tfevents.*       # TensorBoard logs
└── {model_name}_hierarchical_loss/        # HCE models (with suffix)
    └── [same structure as above]
```

### Evaluation Metrics and Files

#### 1. **Overall Performance Metrics** (`{model}_overall_metrics.csv`)
- **Macro F1-score**
- **Micro F1-score**
- **Weighted F1-score**

#### 2. **Per-Class Performance** (`{model}_per_class_metrics.csv`)
- Precision, Recall, F1-score, and Support for each individual cell type

#### 3. **Detailed Stratified Analysis**:
- **`{model}_per_assay_metrics.csv`**: Performance by sequencing technology
- **`{model}_per_dataset_metrics.csv`**: Performance by individual study/dataset
- **`{model}_per_disease_metrics.csv`**: Performance by disease condition
- **`{model}_per_tissue_metrics.csv`**: Performance by specific tissue type
- **`{model}_per_tissue_general_metrics.csv`**: Performance by general tissue category

#### 4. **Raw Predictions** (`{model}_detailed_results.parquet`)
- Complete predictions with cell-level metadata
- Probability distributions over all cell types
- Dataset, tissue, disease, and assay annotations