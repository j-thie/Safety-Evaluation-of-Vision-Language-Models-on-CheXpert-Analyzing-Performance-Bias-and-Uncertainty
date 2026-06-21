# Evaluating-Bias-Detection-Tools-on-Medical-Imaging-Datasets
# Chest X-Ray Vision-Language Model Evaluation Framework

## Overview

This repository contains inference and analysis pipelines for evaluating vision-language models (VLMs) on chest X-ray classification tasks under multiple prompting and information-availability conditions.

The framework supports controlled experiments designed to measure:

* Disease classification performance
* Projection classification performance
* Prompt sensitivity
* Image dependence
* Invalid-response behavior
* Prediction consistency
* Uncertainty across prompts
* Demographic subgroup performance differences

The repository includes scripts for generating model predictions, processing outputs, computing evaluation metrics, and performing statistical analyses.

---

## Features

### Inference Pipelines

Support for multiple vision-language models:

* MedGemma
* Ministral / Mistral

### Experimental Conditions

The framework evaluates model behavior under several controlled settings:

| Condition                  | Description                                                        |
| -------------------------- | ------------------------------------------------------------------ |
| Normal                     | Original chest X-ray provided                                      |
| No Image (Unknown Allowed) | No image provided; model may respond with insufficient information |
| No Image (Known Required)  | No image provided; model must still answer                         |
| Irrelevant Image           | Chest X-ray replaced with an unrelated image                       |

### Evaluation Capabilities

* Accuracy
* Precision
* Recall
* F1 Score
* Macro F1
* Per-class F1
* Specificity
* False Positive Rate (FPR)
* False Negative Rate (FNR)
* Bootstrap confidence intervals
* Pairwise significance testing
* Entropy-based uncertainty measures
* Prediction variability analysis
* Invalid response analysis
* Demographic subgroup comparisons

---

## Repository Structure

### Inference Scripts

#### MedGemma

```text
MG_normal_*
MG_unknown_*
MG_known_*
MG_irrelevant_*
```

Generate predictions using MedGemma under different prompting conditions.

#### Ministral / Mistral

```text
mis_normal_*
mis_unknown_*
mis_known_*
mis_irrelevant_*
```

Equivalent inference pipelines for Ministral-based models.

---

### Cluster Execution

```text
run_medgemma.sh
run_mistral.sh
```

SLURM job submission scripts for GPU clusters.

---

### Performance Evaluation

```text
calculations.py
norm3_f1_acc.py
F1_bootstrapping.py
every_category_per_model.py
```

Compute classification metrics and aggregate results across models and prompts.

---

### Bias and Demographic Analysis

```text
bias.py
invalid_per_gender.py
```

Evaluate subgroup performance differences and invalid response distributions.

---

### Prompt Sensitivity Analysis

```text
per_category_per_prompt_typebootstrapping.py
```

Measures how performance changes across prompt formulations.

---

### Invalid Response Analysis

```text
count_invalids_&_2.py
invlaid_calculations.py
invlaid_calculations3.py
```

Quantifies invalid outputs and evaluates their impact on model performance.

---

### Projection Analysis

```text
count_zeros_for_frontal_lateral.py
everythingforFL.py
f1&accuracy_of_FL.py
```

Analysis of frontal/lateral projection classification tasks.

---

### Uncertainty and Consistency Analysis

```text
entropy_calculations_2.py
entropy_cal_3.py
SD_Comparision.py
SD_Comparision_3.py
```

Measures prediction consistency across prompt variants using entropy and variability metrics.

---

## Dataset Format

The framework expects a JSON dataset structured similarly to:

```json
{
  "image_path": "/path/to/image.png",
  "sex": "Female",
  "patient_id": "patient0001",
  "study_id": "study001",
  "view": "frontal",

  "Cardiomegaly": [
    {
      "question": "Is cardiomegaly present?",
      "answer": 1
    }
  ]
}
```

Each image can contain multiple pathology-specific questions.

---

## Prediction Output Format

Inference scripts produce structured prediction files:

```json
{
  "patient0001": {
    "sex": "Female",
    "images": {
      "image.png": {
        "predictions": [
          {
            "category": "Cardiomegaly",
            "question": "Is cardiomegaly present?",
            "expected_answer": 1,
            "model_answer": 1
          }
        ]
      }
    }
  }
}
```

### Output Labels

| Value   | Meaning                            |
| ------- | ---------------------------------- |
| 0       | Finding absent                     |
| 1       | Finding present                    |
| 2       | Unknown / insufficient information |
| INVALID | Output could not be parsed         |

---

## Installation

### Requirements

```text
Python 3.10+
PyTorch
Transformers
vLLM
NumPy
Pandas
SciPy
Scikit-Learn
Pillow
```

Install dependencies:

```bash
pip install torch transformers vllm numpy pandas scipy scikit-learn pillow
```

---

## Configuration

Before running experiments, update paths inside the scripts:

```python
json_path = "/path/to/dataset.json"
model_dir = "/path/to/model"
output_dir = "/path/to/output"
```

Several scripts contain placeholder paths that must be replaced with local paths.

---

## Running Inference

### MedGemma

```bash
python MG_normal_2.py
```

or submit via SLURM:

```bash
sbatch run_medgemma.sh
```

### Ministral

```bash
python mis_normal_2.py
```

or:

```bash
sbatch run_mistral.sh
```

---

## Running Analysis

### Category-Level Metrics

```bash
python calculations.py
```

### Bootstrap Evaluation

```bash
python F1_bootstrapping.py
```

### Gender-Based Analysis

```bash
python bias.py
```

### Invalid Response Analysis

```bash
python invalid_per_gender.py
```

### Entropy and Uncertainty Analysis

```bash
python entropy_calculations_2.py
python entropy_cal_3.py
```

---

## Evaluation Methodology

### Invalid Predictions

Many evaluation scripts treat invalid outputs as guaranteed incorrect predictions.

For example:

```text
INVALID
```

is converted into an incorrect prediction before metric computation.

This prevents malformed outputs from being ignored during evaluation.

### Bootstrap Confidence Intervals

Several analyses use:

```python
N_BOOT = 10000
SEED = 42
```

to estimate confidence intervals and statistical significance.

---

## Limitations

* Paths must be manually configured before execution.
* The repository assumes a specific dataset structure.
* GPU resources are required for large-scale inference.


