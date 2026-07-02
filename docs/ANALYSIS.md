# Analysis Pipeline

All analysis commands are run from the repository root after inference JSON files have been generated.

The current scripts use hard-coded file lists. Edit the variables described below before running each script. The paths must point to prediction JSON files, not the original CheXpert annotation JSON, unless explicitly stated otherwise.

## Recommended output layout

```text
outputs/
├── predictions/
│   ├── medgemma/
│   └── ministral/
└── analysis/
    ├── metrics/
    ├── bootstrap/
    ├── bias/
    ├── invalid/
    ├── projection/
    └── uncertainty/
```

Most current scripts write into the working directory. Run them from the intended output directory or refactor them to accept an output directory.

## Core metrics

### `calculations.py`

Purpose: calculate per-category accuracy, precision, recall, F1, specificity, and confusion matrices for one prediction file.

Edit:

```python
json_path = "/absolute/path/to/predictions.json"
save_path = "/absolute/path/to/outputs/analysis/metrics"
```

Also replace the hard-coded MedGemma/prompt-F output names:

```python
csv_file = os.path.join(save_path, "<model>_<condition>_<prompt>_metrics.csv")
table_file = os.path.join(save_path, "<model>_<condition>_<prompt>_table.txt")
cm_file = os.path.join(save_path, "<model>_<condition>_<prompt>_confusion_matrices.txt")
```

Run:

```bash
python calculations.py
```

Outputs:

- one metrics CSV;
- one human-readable table;
- one confusion-matrix text file.

Important: this script treats `INVALID` as a guaranteed wrong prediction.

### `norm3_f1_acc.py`

Purpose: calculate three-choice accuracy, macro F1, and per-class F1 for categories across models and prompts.

Edit both lists under:

```python
pathways = {
    "MedGemma": [...],
    "Mistral": [...]
}
```

Run:

```bash
python norm3_f1_acc.py
```

Intended output:

```text
norm3_f1_acc.csv
```

Required fix: remove the second placeholder `raw_data` block, or replace it with:

```python
df = pd.read_csv("norm3_f1_acc.csv")
```

As uploaded, the script writes `norm3_f1_acc.csv` and then attempts to parse placeholder text as CSV.

## Bootstrap analyses

### `F1_bootstrapping.py`

Purpose: patient-level bootstrap confidence intervals for disease and projection F1, model summaries, and pairwise comparisons.

Edit:

```python
FILES = [
    "/absolute/path/to/prompt_a.json",
    "/absolute/path/to/prompt_b.json",
    ...
]
```

Run:

```bash
python F1_bootstrapping.py
```

Outputs:

```text
results_per_file_disease.csv
results_model_summary_disease.csv
results_pairwise_significance_disease.csv
results_per_file_projection.csv
results_model_summary_projection.csv
results_pairwise_significance_projection.csv
```

Required fixes:

1. `parse_name()` assumes exactly three underscore-separated filename components. Replace it with a regular-expression parser or explicit metadata.
2. `paired_bootstrap_diff()` must use the intersection of patient IDs present in both files.
3. Add `zero_division=0` to all `f1_score()` calls.
4. Confirm that every paired file contains the same patients and predictions in the same evaluation condition.

### `every_category_per_model.py`

Purpose: bootstrap category-level F1 and accuracy and aggregate them across prompt files.

Edit:

```python
FILES = [
    "/absolute/path/to/model_condition_prompt_a.json",
    ...
]
```

Run:

```bash
python every_category_per_model.py
```

Output:

```text
final_table_<model>.csv
```

Required fix: replace the fragile `stem.split("_")` filename parser with explicit metadata or a regular expression that supports names such as `MG_normal_3prompt_a.json` and `mis_known_3_prompt_f.json`.

### `per_category_per_prompt_typebootstrapping.py`

Purpose: compare pooled per-category accuracy between MedGemma and Ministral.

Edit:

```python
MED_FILES = [...]
MIS_FILES = [...]
```

Run after fixing:

```bash
python per_category_per_prompt_typebootstrapping.py
```

Intended output:

```text
per_category_bootstrap_results.csv
```

Blocking fix:

```python
MED = pool_by_category(MED_FILES)
MIS = pool_by_category(MIS_FILES)

for category in MED:
    if category not in MIS:
        continue

    med_vals = np.asarray(MED[category])
    mis_vals = np.asarray(MIS[category])
```

The uploaded script creates `MIS_MED` but later references undefined `NONMED` and `nonmed_vals`. It cannot run unchanged.

Also add a fixed NumPy seed, for example:

```python
SEED = 42
rng = np.random.default_rng(SEED)
```

Use the same `rng` object inside both bootstrap functions.

## Demographic analyses

### `bias.py`

Purpose: calculate accuracy, FPR, and FNR by model, prompt, category, and recorded sex.

Set `DATA_FOLDER` to a directory containing prediction JSON files:

```python
DATA_FOLDER = "/absolute/path/to/prediction/json/files"
DATA_PATH = os.path.join(DATA_FOLDER, "*.json")
```

Do not set `DATA_FOLDER` to a JSON filename because the script appends `*.json`.

Run:

```bash
python bias.py
```

Current output: console only.

Recommended change: save the model/sex, prompt/sex, and category/sex result tables as CSV files under `outputs/analysis/bias/`.

Other fixes:

- replace the malformed category-heading string with a normal string;
- remove the unused `chi2_contingency` import or implement and document the intended test;
- use `patient_data.get("sex", "Unknown")`;
- verify that both `Male` and `Female` exist before subtracting columns;
- describe the field as recorded sex unless the dataset documentation explicitly uses another term.

### `invalid_per_gender.py`

Purpose: count invalid responses by category and recorded sex.

Edit:

```python
MEDGEMMA_FILES = [...]
MISTRAL_FILES = [...]
```

Run:

```bash
python invalid_per_gender.py
```

Output:

```text
invalid_summary.csv
```

Required improvement: add rows for prompts with zero invalid responses. The current script creates rows only for categories containing at least one `INVALID`.

## Unknown and invalid response analyses

### `count_invalids_&_2.py`

Purpose: count answer `2`, `INVALID`, and total predictions by scenario, model, and prompt.

Edit all lists in:

```python
EXPERIMENTS = {
    "Normal": {...},
    "No Image Unknown": {...},
    "No Image Known": {...}
}
```

Each model/scenario list should normally contain the six prompt files A-F.

Run with shell quoting:

```bash
python 'count_invalids_&_2.py'
```

Current output: console only.

Recommended rename:

```text
count_unknown_and_invalid.py
```

Recommended change: write the result rows to `unknown_invalid_summary.csv`.

### `invlaid_calculations.py` and `invlaid_calculations3.py`

Purpose: calculate cross-prompt output variability among cases where at least one prompt answer differs from the expected label.

Edit all `EXPERIMENTS` file lists.

Run:

```bash
python invlaid_calculations.py
python invlaid_calculations3.py
```

Current output: console only.

Important interpretation: these scripts do not restrict the analysis to parser value `INVALID`; they include any case with at least one incorrect answer. Rename the metric/documentation to **incorrect-case variability**, or change the filter to:

```python
has_invalid = any(answer == "INVALID" for answer in answers)
```

Required key fix: include the question identity in the comparison key. The current key `(patient_id, image_name, category)` can overwrite multiple questions from the same category. Prefer:

```python
key = (
    patient_id,
    image_name,
    pred["category"],
    pred["question"],
)
```

Correct the filenames:

```text
invalid_calculations.py
invalid_calculations_3choice.py
```

## Prompt uncertainty and consistency

### `entropy_calculations_2.py`

Purpose: normalized Shannon entropy across binary prompt outputs, treating non-`0`/`1` answers as a third class.

Edit all six-prompt file lists inside `EXPERIMENTS`.

Run:

```bash
python entropy_calculations_2.py
```

Current output: console only.

Required fix: compare only keys common to every prompt file:

```python
keys = set.intersection(*(set(result) for result in prompt_results))
```

A more stable key is `(patient_id, image_name, category, question)` rather than `question_idx`, because prompt files may not preserve identical prediction ordering.

### `entropy_cal_3.py`

Purpose: normalized entropy and invalid-response rate for three-choice outputs.

Edit `EXPERIMENTS` and confirm:

```python
SCENARIO_CLASSES = {
    "Normal": 4,
    "No Image Unknown": 4,
    "No Image Known": 4,
}
```

Four classes means `0`, `1`, `2`, and `INVALID`. Use three only if `INVALID` is excluded rather than treated as a class.

Run:

```bash
python entropy_cal_3.py
```

Current output: console only.

Required fixes:

- use common keys across all prompt files;
- replace `question_idx` with category/question identity;
- remove the duplicated full analysis loop;
- save results to a CSV.

### `SD_Comparision.py` and `SD_Comparision_3.py`

Purpose: mean standard deviation of encoded prompt answers.

Edit all `EXPERIMENTS` file lists and run:

```bash
python SD_Comparision.py
python SD_Comparision_3.py
```

Current output: console only.

Required fixes:

- use common keys across prompt files;
- replace `question_idx` with category/question identity;
- document the numeric encoding, because standard deviation on categorical codes depends on the arbitrary class numbers;
- consider keeping entropy as the primary categorical disagreement measure.

Recommended filenames:

```text
prompt_sd_binary.py
prompt_sd_three_choice.py
```

## Projection analyses

### `count_zeros_for_frontal_lateral.py`

Purpose: count predicted zeros, expected zeros, invalid outputs, and total `Frontal_Lateral` questions.

Edit:

```python
FILES = [...]
```

Run:

```bash
python count_zeros_for_frontal_lateral.py
```

Current output: console only.

Recommended change: save one row per prompt to `frontal_lateral_zero_counts.csv`.

### `everythingforFL.py`

Purpose: export record-level frontal/lateral predictions and invalid flags.

Edit:

```python
MEDGEMMA_FILES = [...]
MISTRAL_FILES = [...]
```

Run:

```bash
python everythingforFL.py
```

Output:

```text
frontal_lateral_analysis.csv
```

### `f1&accuracy_of_FL.py`

Purpose: export record-level projection predictions and calculate accuracy/F1 by model, prompt, and view type.

Edit:

```python
MEDGEMMA_FILES = [...]
MISTRAL_FILES = [...]
```

Run with shell quoting:

```bash
python 'f1&accuracy_of_FL.py'
```

Outputs:

```text
frontal_lateral_analysis.csv
frontal_lateral_metrics_detailed.csv
```

Recommended rename:

```text
frontal_lateral_metrics.py
```

Important: this script and `everythingforFL.py` both write `frontal_lateral_analysis.csv`; running the second overwrites the first. Keep only the more complete script or use distinct output paths.

Also confirm that `view_name` actually contains `frontal` or `lateral`. In the prediction schema it may be an image filename rather than the dataset view label. If so, preserve the original `view` metadata during inference and use that field here.

## Dependencies

The model requirements files do not currently include all analysis dependencies. Create a separate pinned file:

```text
analysis_requirements.txt
```

It must include exact tested versions of:

```text
numpy
pandas
scipy
scikit-learn
```

Do not copy arbitrary current versions. Export the versions from the environment used to produce the paper results.

## Recommended execution order

```bash
python calculations.py
python norm3_f1_acc.py
python every_category_per_model.py
python F1_bootstrapping.py
python per_category_per_prompt_typebootstrapping.py
python bias.py
python invalid_per_gender.py
python 'count_invalids_&_2.py'
python invlaid_calculations.py
python invlaid_calculations3.py
python entropy_calculations_2.py
python entropy_cal_3.py
python SD_Comparision.py
python SD_Comparision_3.py
python count_zeros_for_frontal_lateral.py
python everythingforFL.py
python 'f1&accuracy_of_FL.py'
```

Do not run `per_category_per_prompt_typebootstrapping.py` or the final block of `norm3_f1_acc.py` until the blocking defects above are fixed.
