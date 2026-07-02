# Reproducing the Results

## Required order

1. Clone the repository and record the commit.
2. Create the model-specific environment.
3. Obtain CheXpert and prepare `chexpert_qa_long.json`.
4. Obtain MedGemma access/checkpoints.
5. Ensure the Ministral checkpoint can be downloaded or cached.
6. Obtain the irrelevant-image dataset, if required.
7. Configure each inference script.
8. Run all MedGemma conditions and prompt variants.
9. Run all Ministral conditions and prompt variants.
10. Run metric and statistical-analysis scripts.
11. Generate every paper table and figure.
12. Compare against reference results.

## MedGemma inference

```bash
python MG_normal_2.py
python MG_normal_3.py
python MG_unknown_2.py
python MG_unknown_3.py
python MG_known_2.py
python MG_known_3.py
python MG_irrelevant_3.py
```

## Ministral inference

```bash
python mis_normal_2.py
python mis_normal_3.py
python mis_unknown_2.py
python mis_unknown_3.py
python mis_known_2.py
python mis_known_3.py
python mis_irrelevant_3.py
```

## SLURM

```bash
sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="MG_normal_2.py" \
  scripts/run_medgemma.sh

sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="mis_normal_2.py" \
  scripts/run_mistral.sh
```

Repeat for each required script after configuring its embedded paths and prompt.

## Validate each output

Record:

- model and checkpoint revision;
- code commit;
- prompt family and variant;
- condition;
- patient count;
- image count;
- prediction count;
- `0`, `1`, `2`, and `INVALID` counts;
- GPU model;
- peak VRAM;
- runtime;
- random seed, where applicable.


## Analysis commands

Configure the input lists and paths exactly as described in [`ANALYSIS.md`](ANALYSIS.md).

### Core metrics

```bash
python calculations.py
python norm3_f1_acc.py
python every_category_per_model.py
```

### Bootstrap analysis

```bash
python F1_bootstrapping.py
per_category_per_prompt_typebootstrapping.py
python per_category_per_prompt_typebootstrapping.py
```

 

### Demographic and invalid-response analysis

```bash
python bias.py
python invalid_per_gender.py
python 'count_invalids_&_2.py'
python invlaid_calculations.py
python invlaid_calculations3.py
```

### Prompt uncertainty and consistency

```bash
python entropy_calculations_2.py
python entropy_cal_3.py
python SD_Comparision.py
python SD_Comparision_3.py
```

### Projection analysis

```bash
python count_zeros_for_frontal_lateral.py
python everythingforFL.py
python 'f1&accuracy_of_FL.py'
```

The filenames containing `&` must be quoted in shells.

## Analysis outputs to archive

At minimum, archive:

```text
outputs/analysis/
├── metrics/
├── bootstrap/
├── bias/
├── invalid/
├── projection/
└── uncertainty/
```

For every output, record the code commit, input prediction files, model checkpoint, condition, response format, prompt variants, seed, and environment.


