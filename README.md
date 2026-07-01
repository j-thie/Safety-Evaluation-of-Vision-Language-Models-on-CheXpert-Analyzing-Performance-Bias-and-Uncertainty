# Safety Evaluation of Vision-Language Models on CheXpert: Analyzing Performance, Bias, and Uncertainty

Official research code for evaluating MedGemma and Ministral/Mistral on chest X-ray classification under controlled image-availability and prompting conditions.

> **Research use only:** This repository is intended for model evaluation and reproducibility research. It is not a medical device or clinical decision-support system.

## What this repository evaluates

- Disease and projection classification
- Normal, no-image, and irrelevant-image conditions
- Binary and three-choice response formats
- Prompt sensitivity across prompt variants A-F
- Invalid-response behavior
- Prediction consistency and entropy-based uncertainty
- Demographic subgroup performance differences

## Documentation

| Guide | Purpose |
|---|---|
| [Getting started](docs/GETTING_STARTED.md) | Clone-to-first-run instructions |
| [Configuration](docs/CONFIGURATION.md) | Exact variables and locations that must be edited |
| [Inference scripts](docs/INFERENCE.md) | Model/condition command matrix and output names |
| [Prompt conditions](docs/PROMPTS.md) | Prompt families, labels, and variant handling |
| [Reproducing results](docs/REPRODUCING_RESULTS.md) | Required execution order for paper results |
| [Release checks](docs/RELEASE_CHECKS.md) | Code issues that must be resolved before publication |

## Repository layout

```text
.
├── MG_*.py                         # MedGemma inference scripts
├── mis_*.py                        # Ministral/Mistral equivalents
├── run_medgemma.sh                 # MedGemma SLURM launcher
├── run_mistral.sh                  # Ministral/Mistral SLURM launcher
├── MED_environment.yaml            # Python 3.10 Conda base environment
├── MG_requirements.txt             # Pinned MedGemma environment
├── MIS_requirements.txt            # Pinned Ministral/Mistral environment
├── prompt_conditions.pdf           # Full prompt specification
├── docs/                            # Detailed setup and reproduction guides
└── outputs/                         # Generated predictions and analyses
```

## Getting started

All commands are run from the repository root.

### 1. Clone the repository

```bash
git clone <FINAL_REPOSITORY_URL>
cd Safety-Evaluation-of-Vision-Language-Models-on-CheXpert-Analyzing-Performance-Bias-and-Uncertainty
```

### 2. Create the MedGemma environment

```bash
conda env create -f MED_environment.yaml
conda activate medgemma_paper
python -m pip install --upgrade pip
python -m pip install -r MG_requirements.txt
```

The supplied Conda file defines Python `3.10`; the requirements file pins the Python packages used in the exported environment.

### 3. Create the Ministral/Mistral environment

```bash
conda create -n ministral_paper -c conda-forge python=3.10 libstdcxx-ng pip
conda activate ministral_paper
python -m pip install --upgrade pip
python -m pip install -r MIS_requirements.txt
```

Before release, replace the non-portable local `packaging @ file:///...` entry in `MIS_requirements.txt` with a normal pinned package version.

### 4. Obtain the data

Download CheXpert from the official Stanford AIMI dataset page and comply with its access and citation terms:

<https://aimi.stanford.edu/datasets/chexpert-chest-x-rays>

The scripts expect a JSON list whose records include an `image_path`, patient metadata, and pathology-specific question lists. See [Getting started](docs/GETTING_STARTED.md#3-place-the-data).

### 5. Configure the scripts

The current scripts do not accept command-line path arguments. Edit these variables directly before running:

```python
PROMPT_NAME = "prompt_a"
json_path = "/absolute/path/to/chexpert_qa_long.json"
model_dir = "/absolute/path/to/model"
output_dir = "/absolute/path/to/output"
```

The irrelevant-image condition also requires:

```python
imagenet_root = Path("/absolute/path/to/irrelevant/image/dataset")
```

Exact files and current line locations are listed in [Configuration](docs/CONFIGURATION.md).

### 6. Run inference

MedGemma examples:

```bash
python MG_normal_2.py
python MG_normal_3.py
python MG_unknown_2.py
python MG_unknown_3.py
python MG_known_2.py
python MG_known_3.py
python MG_irrelevant_3.py
```

Ministral/Mistral uses the corresponding `mis_*.py` scripts:

```bash
python mis_normal_2.py
python mis_normal_3.py
python mis_unknown_2.py
python mis_unknown_3.py
python mis_known_2.py
python mis_known_3.py
python mis_irrelevant_3.py
```

Only run filenames that actually exist in the repository. See [Inference scripts](docs/INFERENCE.md) for output names and condition details.

### 7. Run on SLURM

The uploaded launcher requires correction before use. A corrected template is provided at [`scripts/run_medgemma.example.sh`](scripts/run_medgemma.example.sh).

Example:

```bash
sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="MG_normal_2.py" \
  scripts/run_medgemma.example.sh
```

The Python scripts still control their own dataset, model, and output paths, so configure those first.

## Model settings currently encoded in MedGemma scripts

- Model class: `AutoModelForImageTextToText`
- Processor: `AutoProcessor`
- Precision: `torch.bfloat16`
- Device placement: `device_map="auto"`
- Sampling: disabled with `do_sample=False`
- Maximum generated tokens: `32` in the uploaded image-input scripts
- Output format: JSON grouped by patient and image

## Hardware

The supplied MedGemma SLURM file requests:

| Resource | Requested value |
|---|---:|
| GPU | 1 H100-class GPU partition |
| CPUs | 8 |
| System memory | 180 GB |
| Wall time | 1 hour |

This is a **tested/requested cluster configuration, not a measured minimum**. Before publication, record peak VRAM and actual runtime for every model and condition.

## Output labels

| Value | Meaning |
|---|---|
| `0` | Finding absent |
| `1` | Finding present |
| `2` | Insufficient information |
| `INVALID` | Output could not be parsed |

## Citation and license

Add the final paper citation, repository license, and model/dataset license notices before release.
