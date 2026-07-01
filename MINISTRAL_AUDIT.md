# Ministral Inference Audit

## Exact runtime configuration

All supplied scripts load:

```python
LLM(
    model="mistralai/Ministral-3-14B-Instruct-2512",
    tokenizer="mistralai/Ministral-3-14B-Instruct-2512",
    tokenizer_mode="mistral",
    gpu_memory_utilization=0.95,
    max_model_len=32000,
)
```

They use:

```python
SamplingParams(
    max_tokens=32,
    temperature=0.0,
    top_p=1.0,
)
```

## Script summary

| Script | Condition | Labels | Image passed |
|---|---|---|---:|
| `mis_normal_2.py` | Normal | `0`, `1` | Yes |
| `mis_normal_3.py` | Normal | `0`, `1`, `2` | Yes |
| `mis_unknown_2.py` | No image, absence undisclosed | `0`, `1` | No |
| `mis_unknown_3.py` | No image, absence undisclosed | `0`, `1`, `2` | No |
| `mis_known_2.py` | No image, absence disclosed | `0`, `1` | No |
| `mis_known_3.py` | No image, absence disclosed | `0`, `1`, `2` | No |
| `mis_irrelevant_3.py` | Random ImageNet image | `0`, `1`, `2` | Yes |

## Required edits before each run

- `PROMPT_NAME`
- `json_path`
- `output_dir`
- `imagenet_root` for the irrelevant-image condition
- model/tokenizer ID if a different checkpoint is required

## Main reproducibility risks

- Prompt selection is manual.
- No-image scripts still require and load image files.
- Known-condition scripts contain literal escaped newline sequences.
- Irrelevant-image sampling is unseeded and does not preserve source identifiers.
- Output filename formatting is inconsistent.
- The original SLURM launcher passes unsupported arguments.
