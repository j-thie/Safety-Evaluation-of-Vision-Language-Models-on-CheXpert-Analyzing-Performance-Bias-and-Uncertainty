# Configuration

The uploaded inference scripts use hard-coded source variables rather than command-line arguments. File path plus variable name is authoritative; line numbers refer to the supplied versions.

## MedGemma configuration map

| Script | `PROMPT_NAME` | `json_path` | `model_dir` | `imagenet_root` | `output_dir` |
|---|---:|---:|---:|---:|---:|
| `MG_normal_2.py` | 62 | 78 | 87 | - | 237 |
| `MG_normal_3.py` | 67 | 83 | 92 | - | 241 |
| `MG_irrelevant_3.py` | 67 | 82 | 94 | 88 | 242 |
| `MG_unknown_2.py` | 61 | 77 | 86 | - | 233 |
| `MG_unknown_3.py` | 68 | 75 | 84 | - | 222 |
| `MG_known_2.py` | 67 | 74 | 83 | - | 218 |
| `MG_known_3.py` | 69 | 76 | 85 | - | 222 |

## Ministral configuration map

| Script | `PROMPT_NAME` | `json_path` | Model/tokenizer | `imagenet_root` | `output_dir` |
|---|---:|---:|---:|---:|---:|
| `mis_normal_2.py` | 62 | 74 | 85-86 | - | 191 |
| `mis_normal_3.py` | 69 | 81 | 92-93 | - | 197 |
| `mis_unknown_2.py` | 87 | 99 | 110-111 | - | 212 |
| `mis_unknown_3.py` | 90 | 102 | 113-114 | - | 216 |
| `mis_known_2.py` | 87 | 99 | 110-111 | - | 213 |
| `mis_known_3.py` | 91 | 103 | 114-115 | - | 218 |
| `mis_irrelevant_3.py` | 71 | 83 | 97-98 | 89 | 203 |

Ministral uses:

```python
llm = LLM(
    model="mistralai/Ministral-3-14B-Instruct-2512",
    tokenizer="mistralai/Ministral-3-14B-Instruct-2512",
    tokenizer_mode="mistral",
    gpu_memory_utilization=0.95,
    max_model_len=32000,
)

sampling_params = SamplingParams(
    max_tokens=32,
    temperature=0.0,
    top_p=1.0,
)
```

## Prompt configuration

Every prompt is embedded directly in the script.

Changing:

```python
PROMPT_NAME = "prompt_f"
```

changes only the output filename. It does not select prompt F.

To reproduce variants A-F correctly:

1. edit the embedded prompt text;
2. set the matching `PROMPT_NAME`;
3. confirm the parser accepts the labels advertised by the prompt; and
4. use a separate output directory or filename for every run.

## No-image behavior

The `known` and `unknown` Ministral scripts:

- read `entry["image_path"]`;
- require that the file exists;
- load and base64-encode the image;
- derive the patient and image identifiers from it;
- but do not include the image in the `messages` passed to `llm.chat`.

This means they are operationally no-image conditions, but they still require the image files to exist.

## SLURM settings

Both supplied launchers request:

```bash
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
```

### Original launcher issues

The original launchers:

- invoke placeholder script paths;
- pass `--input_folder`, `--output_folder`, and `--batch_size`, which the inference scripts do not parse;
- define an `OUTPUT_DIR` that is not consumed by the inference script;
- rely on environment names that differ from the documentation package.

Use the corrected templates:

```text
scripts/run_medgemma.example.sh
scripts/run_mistral.example.sh
```

## Search for unreplaced paths

```bash
grep -RInE \
  '(/home/|/scratch/|/work/|/mnt/|/data/|/path/to/|/pathway/to/|/load/|/add/wanted/)' \
  --include='*.py' --include='*.sh' .
```
