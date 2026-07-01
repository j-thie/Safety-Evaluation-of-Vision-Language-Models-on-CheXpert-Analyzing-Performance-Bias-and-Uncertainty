# Getting Started

## 1. Clone and record the code version

```bash
git clone <FINAL_REPOSITORY_URL>
cd Safety-Evaluation-of-Vision-Language-Models-on-CheXpert-Analyzing-Performance-Bias-and-Uncertainty
git rev-parse HEAD
```

Save the commit hash with the experiment outputs.

## 2. Create the environments

### MedGemma

```bash
conda env create -f MED_environment.yaml
conda activate medgemma_paper
python -m pip install --upgrade pip
python -m pip install -r MG_requirements.txt
```

### Ministral

```bash
conda create -n ministral_paper -c conda-forge python=3.10 libstdcxx-ng pip
conda activate ministral_paper
python -m pip install --upgrade pip
python -m pip install -r MIS_requirements.txt
```

Important versions found in the supplied files:

| Component | MedGemma | Ministral |
|---|---:|---:|
| Python | 3.10 | 3.10 |
| PyTorch | 2.10.0 | 2.10.0 |
| torchvision | 0.25.0 | 0.25.0 |
| Transformers | 4.57.6 | 4.57.6 |
| Accelerate | 1.13.0 | 1.13.0 |
| vLLM | 0.18.1 | 0.19.0 |
| Pillow | 12.1.1 | 12.1.1 |
| NumPy | 2.2.6 | 2.2.6 |

The Ministral requirements currently contain a local `packaging @ file:///...` entry. Replace it with a portable pinned version before release.

Validate the environment:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
PY
```

## 3. Place the data

Obtain CheXpert from:

<https://aimi.stanford.edu/datasets/chexpert-chest-x-rays>

Recommended layout:

```text
data/
├── CheXpert-v1.0/
└── evaluation/
    └── chexpert_qa_long.json
```

Each record is expected to resemble:

```json
{
  "image_path": "/absolute/path/to/image.png",
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

Check image paths:

```bash
python - <<'PY'
import json
from pathlib import Path

json_path = Path("data/evaluation/chexpert_qa_long.json")
entries = json.loads(json_path.read_text())
missing = [e["image_path"] for e in entries if not Path(e["image_path"]).exists()]

print("Entries:", len(entries))
print("Missing images:", len(missing))
for path in missing[:10]:
    print(path)
PY
```

The current no-image scripts still check and load each original image even though the image is not passed to the model. Therefore, the source image files must remain available unless those scripts are refactored.

## 4. Configure a script

Edit the intended script directly.

### MedGemma

```python
PROMPT_NAME = "prompt_a"
json_path = "/data/evaluation/chexpert_qa_long.json"
model_dir = "/models/medgemma"
output_dir = "/results/medgemma/normal_binary"
```

### Ministral

```python
PROMPT_NAME = "prompt_a"
json_path = "/data/evaluation/chexpert_qa_long.json"
output_dir = "/results/ministral/normal_binary"
```

The Ministral model ID is hard-coded in each script:

```python
model="mistralai/Ministral-3-14B-Instruct-2512"
tokenizer="mistralai/Ministral-3-14B-Instruct-2512"
```

For irrelevant-image experiments:

```python
imagenet_root = Path("/data/imagenet")
```

`PROMPT_NAME` changes the output filename only. It does not select the prompt template. Verify that the embedded prompt and `PROMPT_NAME` match.

## 5. Create a smoke-test dataset

The current scripts process the complete JSON. Create a temporary one-record file:

```bash
python - <<'PY'
import json
from pathlib import Path

source = Path("data/evaluation/chexpert_qa_long.json")
target = Path("data/evaluation/smoke_test.json")

entries = json.loads(source.read_text())
target.write_text(json.dumps(entries[:1], indent=2))
print(target)
PY
```

Temporarily set `json_path` to this file.

MedGemma smoke test:

```bash
python MG_normal_2.py
```

Ministral smoke test:

```bash
python mis_normal_2.py
```

## 6. Run all conditions

See [`INFERENCE.md`](INFERENCE.md) for the complete command matrix and output filenames.

## 7. Validate outputs

```bash
python -m json.tool /path/to/generated-output.json > /dev/null
```

For every output, record:

- patients;
- images;
- predictions;
- counts of `0`, `1`, `2`, and `INVALID`;
- prompt variant;
- model revision;
- code commit;
- GPU and peak VRAM;
- runtime.

## 8. Run the analysis pipeline

The analysis scripts still need to be audited. Add their exact inputs, commands, outputs, and paper table/figure mappings to [`REPRODUCING_RESULTS.md`](REPRODUCING_RESULTS.md).
