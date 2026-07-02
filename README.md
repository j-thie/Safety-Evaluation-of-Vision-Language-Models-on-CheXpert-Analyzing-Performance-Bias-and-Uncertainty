# Safety Evaluation of Vision-Language Models on CheXpert: Analyzing Performance, Bias, and Uncertainty
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Models](https://img.shields.io/badge/Models-MedGemma%20%7C%20Ministral-green)
![Dataset](https://img.shields.io/badge/Dataset-CheXpert-yellow)
![GPU Required](https://img.shields.io/badge/GPU-Required-red)
![Tested on bwHPC](https://img.shields.io/badge/Tested%20on-bwHPC-blueviolet)
![Research Use](https://img.shields.io/badge/Use-Research%20Only-orange)

Research code for evaluating MedGemma and Ministral on chest X-ray classification under controlled image-availability and prompting conditions.

> **Research use only:** This repository is intended for model-evaluation and reproducibility research. It is not a medical device or clinical decision-support system.

## What this repository evaluates

- Disease and projection classification
- Original-image, no-image, and irrelevant-image conditions
- Binary (`0`/`1`) and three-choice (`0`/`1`/`2`) responses
- Prompt sensitivity across variants A-F
- Invalid-response behavior
- Prediction consistency and entropy-based uncertainty
- Demographic subgroup performance differences

## Documentation

| Guide | Purpose |
|---|---|
| [Getting started](docs/GETTING_STARTED.md) | Clone-to-first-run instructions |
| [Configuration](docs/CONFIGURATION.md) | Exact variables and line locations to edit |
| [Inference scripts](docs/INFERENCE.md) | Model/condition commands and output names |
| [Prompt conditions](docs/PROMPTS.md) | Prompt families, labels, and variants |
| [Reproducing results](docs/REPRODUCING_RESULTS.md) | Required execution order for paper results |
| [Release checks](docs/RELEASE_CHECKS.md) | Issues to resolve before publication |

## Repository layout

```text
.
├── MG_*.py                         # MedGemma inference scripts
├── mis_*.py                        # Ministral inference scripts
├── run_medgemma.sh                 # Original MedGemma SLURM launcher
├── run_mistral.sh                  # Original Ministral SLURM launcher
├── MED_environment.yaml            # Python 3.10 Conda base environment
├── MG_requirements.txt             # Pinned MedGemma packages
├── MIS_requirements.txt            # Pinned Ministral packages
├── prompt_conditions.pdf           # Complete prompt specification
├── docs/                           # Detailed setup and reproduction guides
├── scripts/                        # Corrected launcher templates
└── outputs/                        # Generated predictions and analyses
```

## Getting started

All commands below are run from the repository root.

### 1. Clone the repository

```bash
git clone (https://github.com/j-thie/Safety-Evaluation-of-Vision-Language-Models-on-CheXpert-Analyzing-Performance-Bias-and-Uncertainty.git)
cd Safety-Evaluation-of-Vision-Language-Models-on-CheXpert-Analyzing-Performance-Bias-and-Uncertainty
```

### 2. Create the MedGemma environment

```bash
conda env create -f MED_environment.yaml
conda activate medgemma
python -m pip install --upgrade pip
python -m pip install -r MG_requirements.txt
```

### 3. Create the Ministral environment

```bash
conda create -n ministral_paper -c conda-forge python=3.10 libstdcxx-ng pip
conda activate ministral
python -m pip install --upgrade pip
python -m pip install -r MIS_requirements.txt
```



### 4. Obtain CheXpert

Request CheXpert through the official Stanford AIMI dataset page and follow its access and citation terms:

<https://aimi.stanford.edu/datasets/chexpert-chest-x-rays>

The scripts expect a JSON list containing image paths, patient metadata, and pathology-specific question lists. See [Getting started](docs/GETTING_STARTED.md#3-place-the-data).

### 5. Configure paths


MedGemma:

```python
PROMPT_NAME = "prompt_a"
json_path = "/absolute/path/to/chexpert_qa_long.json"
model_dir = "/absolute/path/to/medgemma"
output_dir = "/absolute/path/to/output"
```

Ministral:

```python
PROMPT_NAME = "prompt_a"
json_path = "/absolute/path/to/chexpert_qa_long.json"
output_dir = "/absolute/path/to/output"
```

The Ministral checkpoint is currently hard-coded as:

```python
model = "mistralai/Ministral-3-14B-Instruct-2512"
tokenizer = "mistralai/Ministral-3-14B-Instruct-2512"
```

Irrelevant-image scripts additionally require:

```python
imagenet_root = Path("/absolute/path/to/irrelevant/image/dataset")
```

Exact file and line locations are listed in [Configuration](docs/CONFIGURATION.md).

### 6. Run inference
<table>
  <thead>
    <tr>
      <th>MedGemma</th>
      <th>Ministral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <code>python MG_normal_2.py</code><br>
        <code>python MG_normal_3.py</code><br>
        <code>python MG_unknown_2.py</code><br>
        <code>python MG_unknown_3.py</code><br>
        <code>python MG_known_2.py</code><br>
        <code>python MG_known_3.py</code><br>
        <code>python MG_irrelevant_3.py</code>
      </td>
      <td>
        <code>python mis_normal_2.py</code><br>
        <code>python mis_normal_3.py</code><br>
        <code>python mis_unknown_2.py</code><br>
        <code>python mis_unknown_3.py</code><br>
        <code>python mis_known_2.py</code><br>
        <code>python mis_known_3.py</code><br>
        <code>python mis_irrelevant_3.py</code>
      </td>
    </tr>
  </tbody>
</table>


The naming convention is:

- `normal`: original chest X-ray is supplied;
- `unknown`: the image is withheld without telling the model;
- `known`: the image is withheld and the model is explicitly told it is missing;
- `irrelevant`: an unrelated ImageNet image is supplied;
- `_2`: binary response (`0` or `1`);
- `_3`: three-choice response (`0`, `1`, or `2`).

### 7. Run on SLURM

Corrected launcher templates are included:

```bash
sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="MG_normal_2.py" \
  scripts/run_medgemma.sh

sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="mis_normal_2.py" \
  scripts/run_mistral.sh
```

The Python scripts still control their dataset, model, prompt, irrelevant-image, and output settings, so configure them before submitting.

## Inference settings


| MedGemma                                                                                                                                | Ministral                                                                                                                                                                                                          |
| --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `AutoModelForImageTextToText`<br>`AutoProcessor`<br>`torch.bfloat16`<br>`device_map="auto"`<br>`do_sample=False`<br>`max_new_tokens=32` | Model: `mistralai/Ministral-3-14B-Instruct-2512`<br>Runtime: `vLLM`<br>`tokenizer_mode="mistral"`<br>`gpu_memory_utilization=0.95`<br>`max_model_len=32000`<br>`max_tokens=32`<br>`temperature=0.0`<br>`top_p=1.0` |


## Hardware

Both supplied SLURM launchers request:

| Resource | Requested value |
|---|---:|
| GPU | 1 GPU on an H100 partition |
| CPUs | 8 |
| System memory | 180 GB |
| Wall time | 1 hour |



## Output labels

| Value | Meaning |
|---|---|
| `0` | Finding absent |
| `1` | Finding present |
| `2` | Insufficient information |
| `INVALID` | Output could not be parsed |

## Citation and license

The authors acknowledge support by the state of Baden-Württemberg through bwHPC.
