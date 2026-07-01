# Reproducing the Results

## Required order

1. Clone the repository and record the commit.
2. Create the model-specific environment.
3. Obtain CheXpert and prepare `chexpert_qa_long.json`.
4. Obtain MedGemma access/checkpoints.
5. Ensure the Ministral checkpoint can be downloaded or cached.
6. Obtain the irrelevant-image dataset, if required.
7. Configure each inference script.
8. Run one-record smoke tests.
9. Run all MedGemma conditions and prompt variants.
10. Run all Ministral conditions and prompt variants.
11. Validate outputs.
12. Run metric and statistical-analysis scripts.
13. Generate every paper table and figure.
14. Compare against reference results.

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
  scripts/run_medgemma.example.sh

sbatch \
  --export=ALL,REPO_DIR="$PWD",SCRIPT="mis_normal_2.py" \
  scripts/run_mistral.example.sh
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

The analysis scripts have not yet been supplied for this audit.

| Paper result | Script | Required inputs | Command | Output |
|---|---|---|---|---|
| Overall metrics | `[ADD]` | `[ADD]` | `[ADD]` | `[ADD]` |
| Bootstrap confidence intervals | `[ADD]` | `[ADD]` | `[ADD]` | `[ADD]` |
| Demographic subgroup analysis | `[ADD]` | `[ADD]` | `[ADD]` | `[ADD]` |
| Invalid-response analysis | `[ADD]` | `[ADD]` | `[ADD]` | `[ADD]` |
| Projection analysis | `[ADD]` | `[ADD]` | `[ADD]` | `[ADD]` |
| Entropy/consistency analysis | `[ADD]` | `[ADD]` | `[ADD]` | `[ADD]` |
| Figure 2 | `[ADD]` | `[ADD]` | `[ADD]` | `assets/figures/figure_2.*` |
| Figure 3 | `[ADD]` | `[ADD]` | `[ADD]` | `assets/figures/figure_3.*` |

## Reference results

| Check | Expected |
|---|---:|
| Patients | `[ADD]` |
| Images | `[ADD]` |
| Questions | `[ADD]` |
| MedGemma normal macro F1 | `[ADD]` |
| Ministral normal macro F1 | `[ADD]` |
| Invalid-response rate | `[ADD]` |
