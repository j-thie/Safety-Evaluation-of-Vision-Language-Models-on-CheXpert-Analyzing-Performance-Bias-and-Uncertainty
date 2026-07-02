# Configuration

The inference and analysis scripts currently use variables defined directly inside the Python files. They do not accept command-line arguments for dataset paths, model paths, or output directories.

Before running a script, open it and replace the placeholder values described below.

## Inference configuration

### MedGemma

The MedGemma scripts use the following configuration variables:

```python
PROMPT_NAME = "prompt_a"

json_path = "/absolute/path/to/chexpert_qa_long.json"

model_dir = "/absolute/path/to/medgemma/model"

output_dir = "/absolute/path/to/output/directory"
```

Update them as follows:

| Variable      | Required value                                                    |
| ------------- | ----------------------------------------------------------------- |
| `PROMPT_NAME` | Name of the prompt variant used, such as `prompt_a`               |
| `json_path`   | Absolute path to the prepared CheXpert question-answer JSON file  |
| `model_dir`   | Local MedGemma checkpoint directory or supported model identifier |
| `output_dir`  | Directory where the generated prediction JSON will be saved       |

These variables appear in:

```text
MG_normal_2.py
MG_normal_3.py
MG_unknown_2.py
MG_unknown_3.py
MG_known_2.py
MG_known_3.py
MG_irrelevant_3.py
```

The irrelevant-image condition additionally requires:

```python
imagenet_root = Path("/absolute/path/to/irrelevant/image/dataset")
```

`imagenet_root` must contain the unrelated images used in the irrelevant-image experiment.

### MedGemma example

```python
PROMPT_NAME = "prompt_a"

json_path = "/data/chexpert/evaluation/chexpert_qa_long.json"

model_dir = "/models/medgemma"

output_dir = "/results/medgemma/normal/binary"
```

---

### Ministral

The Ministral scripts use:

```python
PROMPT_NAME = "prompt_a"

json_path = "/absolute/path/to/chexpert_qa_long.json"

output_dir = "/absolute/path/to/output/directory"
```

These variables appear in:

```text
mis_normal_2.py
mis_normal_3.py
mis_unknown_2.py
mis_unknown_3.py
mis_known_2.py
mis_known_3.py
mis_irrelevant_3.py
```

The Ministral model is currently defined directly inside each script:

```python
llm = LLM(
    model="mistralai/Ministral-3-14B-Instruct-2512",
    tokenizer="mistralai/Ministral-3-14B-Instruct-2512",
    tokenizer_mode="mistral",
    gpu_memory_utilization=0.95,
    max_model_len=32000,
)
```

Generation settings are:

```python
sampling_params = SamplingParams(
    max_tokens=32,
    temperature=0.0,
    top_p=1.0,
)
```

The irrelevant-image script additionally requires:

```python
imagenet_root = Path("/absolute/path/to/irrelevant/image/dataset")
```

### Ministral example

```python
PROMPT_NAME = "prompt_a"

json_path = "/data/chexpert/evaluation/chexpert_qa_long.json"

output_dir = "/results/ministral/normal/binary"
```

---

## Prompt configuration

Prompt text is embedded directly inside each inference script.

Changing:

```python
PROMPT_NAME = "prompt_f"
```

changes only the generated output filename. It does not automatically select prompt F.

For every inference run:

1. Replace the embedded prompt text with the intended prompt variant.
2. Set `PROMPT_NAME` to the matching prompt name.

For example, a three-choice prompt must use a parser that accepts:

```text
0
1
2
```

A binary prompt must use a parser that accepts only:

```text
0
1
```

The complete prompt definitions are provided in [`PROMPTS.md`](PROMPTS.md).

---

## No-image conditions

The no-image conditions contain two different experiment types.

### No Image — Known

The model is explicitly told that the chest X-ray is unavailable.

Relevant scripts include:

```text
MG_known_2.py
MG_known_3.py
mis_known_2.py
mis_known_3.py
```

### No Image — Unknown

The image is not supplied to the model, but the prompt does not explicitly state that it is unavailable.

Relevant scripts include:

```text
MG_unknown_2.py
MG_unknown_3.py
mis_unknown_2.py
mis_unknown_3.py
```

The current Ministral no-image scripts still:

* read `entry["image_path"]`;
* check that the original image exists;
* load and encode the image;
* derive patient and image identifiers from the path.

However, the encoded image is not included in the message sent to `llm.chat`.

The condition is therefore operationally image-free, but the original image files must still be available. This should be simplified in a future code revision.

---

## Irrelevant-image configuration

The irrelevant-image scripts replace the CheXpert image with an unrelated image.

Configure:

```python
imagenet_root = Path("/absolute/path/to/irrelevant/image/dataset")
```

The current scripts recursively search for:

```text
*.JPEG
```

Before reproducing the experiment, also confirm:

* which unrelated-image dataset was used;
* whether its license permits this use;
* whether sampling was performed with replacement;
* which random seed was used; and
* whether the selected irrelevant-image mapping was saved.

A fixed random seed should be added for reproducibility:

```python
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
```

---

## Output directories

Create output directories before running inference, or confirm that the scripts create them automatically.

Recommended layout:

```text
outputs/
├── predictions/
│   ├── medgemma/
│   │   ├── normal/
│   │   ├── no_image_known/
│   │   ├── no_image_unknown/
│   │   └── irrelevant_image/
│   └── ministral/
│       ├── normal/
│       ├── no_image_known/
│       ├── no_image_unknown/
│       └── irrelevant_image/
└── analysis/
    ├── metrics/
    ├── bootstrap/
    ├── bias/
    ├── invalid_responses/
    ├── uncertainty/
    └── projection/
```



## SLURM configuration

The supplied MedGemma and Ministral launchers request:

```bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
```

These values describe the cluster configuration used or requested during development. They should not be presented as minimum hardware requirements unless they have been measured.

Before submitting a job, update:

```bash
REPO_DIR="/absolute/path/to/repository"
SCRIPT="MG_normal_2.py"
```

For Ministral:

```bash
REPO_DIR="/absolute/path/to/repository"
SCRIPT="mis_normal_2.py"
```

Submit the corrected launchers using:

```bash
sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="MG_normal_2.py" \
  scripts/run_medgemma.sh
```

```bash
sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="mis_normal_2.py" \
  scripts/run_mistral.sh
```

The dataset, model, prompt, and output settings are still read from inside the selected Python script.

### Original launcher problems

The original launchers should not be used unchanged because they:

* invoke placeholder script paths;
* pass unsupported arguments such as `--input_folder`;
* define an `OUTPUT_DIR` that is not read by the Python scripts; and
* use environment names that may not match the provided environment files.

Use the corrected launchers stored in:

```text
scripts/run_medgemma.sh
scripts/run_mistral.sh
```

---
## Analysis configuration map

| Script | Variable(s) to edit | Input type | Output |
|---|---|---|---|
| `calculations.py` | `json_path`, `save_path`, three output filenames | one prediction JSON | CSV and text files |
| `norm3_f1_acc.py` | `pathways` | MedGemma and Ministral prompt JSONs | `norm3_f1_acc.csv` |
| `every_category_per_model.py` | `FILES` | all prompt JSONs | `final_table_<model>.csv` |
| `F1_bootstrapping.py` | `FILES` | matched prompt JSONs | six bootstrap CSV files |
| `per_category_per_prompt_typebootstrapping.py` | `MED_FILES`, `MIS_FILES` | model prompt JSONs | `per_category_bootstrap_results.csv` |
| `bias.py` | `DATA_FOLDER` | directory of prediction JSONs | console unless modified |
| `invalid_per_gender.py` | `MEDGEMMA_FILES`, `MISTRAL_FILES` | model prompt JSONs | `invalid_summary.csv` |
| `count_invalids_&_2.py` | `EXPERIMENTS` | six prompts per scenario/model | console unless modified |
| `invlaid_calculations.py` | `EXPERIMENTS` | binary prompt JSONs | console unless modified |
| `invlaid_calculations3.py` | `EXPERIMENTS` | three-choice prompt JSONs | console unless modified |
| `entropy_calculations_2.py` | `EXPERIMENTS` | binary prompt JSONs | console unless modified |
| `entropy_cal_3.py` | `EXPERIMENTS`, `SCENARIO_CLASSES` | three-choice prompt JSONs | console unless modified |
| `SD_Comparision.py` | `EXPERIMENTS` | binary prompt JSONs | console unless modified |
| `SD_Comparision_3.py` | `EXPERIMENTS` | three-choice prompt JSONs | console unless modified |
| `count_zeros_for_frontal_lateral.py` | `FILES` | projection prompt JSONs | console unless modified |
| `everythingforFL.py` | `MEDGEMMA_FILES`, `MISTRAL_FILES` | projection prompt JSONs | record-level CSV |
| `f1&accuracy_of_FL.py` | `MEDGEMMA_FILES`, `MISTRAL_FILES` | projection prompt JSONs | record and metrics CSVs |

See [`ANALYSIS.md`](ANALYSIS.md) for exact commands and blocking fixes.
