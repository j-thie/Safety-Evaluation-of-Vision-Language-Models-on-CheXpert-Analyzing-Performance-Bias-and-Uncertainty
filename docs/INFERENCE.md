# Inference Scripts

## Condition naming

| Name | Meaning |
|---|---|
| `normal` | Original chest X-ray is supplied |
| `unknown` | Image is withheld, but the prompt does not disclose that it is missing |
| `known` | Image is withheld and the prompt explicitly states that it is missing |
| `irrelevant` | An unrelated ImageNet image is supplied |
| `_2` | Binary response: `0` or `1` |
| `_3` | Three-choice response: `0`, `1`, or `2` |

## Inference command matrix

The `_2` scripts use binary outputs (`0` or `1`), while the `_3` scripts allow three-choice outputs (`0`, `1`, or `2`).

| Model     | Condition          | Script                |     Image sent to model | Parser accepts | Current prompt name | Output filename                 |
| --------- | ------------------ | --------------------- | ----------------------: | -------------- | ------------------- | ------------------------------- |
| MedGemma  | Normal             | `MG_normal_2.py`      |                     Yes | `0`, `1`       | `prompt_f`          | `MG_normal_2prompt_f.json`      |
| MedGemma  | Normal             | `MG_normal_3.py`      |                     Yes | `0`, `1`, `2`  | `prompt_f`          | `MG_normal_3prompt_f.json`      |
| MedGemma  | No Image — Unknown | `MG_unknown_2.py`     |                      Yes| `0`, `1`       | `prompt_f`          | `MG_unknown_2prompt_f.json`     |
| MedGemma  | No Image — Unknown | `MG_unknown_3.py`     |                      No | `0`, `1`, `2`  | `prompt_f`          | `MG_unknown_3prompt_f.json`     |
| MedGemma  | No Image — Known   | `MG_known_2.py`       |                      No | `0`, `1`       | `prompt_f`          | `MG_known_2prompt_f.json`       |
| MedGemma  | No Image — Known   | `MG_known_3.py`       |                      No | `0`, `1`, `2`  | `prompt_f`          | `MG_known_3prompt_f.json`       |
| MedGemma  | Irrelevant Image   | `MG_irrelevant_3.py`  |    Random ImageNet JPEG | `0`, `1`, `2`  | `prompt_e`          | `MG_irrelevant_3prompt_e.json`  |
| Ministral | Normal             | `mis_normal_2.py`     |                     Yes | `0`, `1`       | `prompt_f`          | `mis_normal_2prompt_f.json`     |
| Ministral | Normal             | `mis_normal_3.py`     |                     Yes | `0`, `1`, `2`  | `prompt_a`          | `mis_normal_3prompt_a.json`     |
| Ministral | No Image — Unknown | `mis_unknown_2.py`    |                      No | `0`, `1`       | `prompt_f`          | `mis_unknown_2prompt_f.json`    |
| Ministral | No Image — Unknown | `mis_unknown_3.py`    |                      No | `0`, `1`, `2`  | `prompt_f`          | `mis_unknown_3prompt_f.json`    |
| Ministral | No Image — Known   | `mis_known_2.py`      |                      No | `0`, `1`       | `prompt_f`          | `mis_known_2_prompt_f.json`     |
| Ministral | No Image — Known   | `mis_known_3.py`      |                      No | `0`, `1`, `2`  | `prompt_f`          | `mis_known_3_prompt_f.json`     |
| Ministral | Irrelevant Image   | `mis_irrelevant_3.py` |    Random ImageNet JPEG | `0`, `1`, `2`  | `prompt_f`          | `mis_irrelevant_3prompt_f.json` |

> The values under **Current prompt name** describe the uploaded script versions. Changing `PROMPT_NAME` changes only the output filename; it does not automatically change the embedded prompt text.

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


See [`CONFIGURATION.md`](CONFIGURATION.md) for the variables that must be edited before running each script.

Run:

```bash
python mis_normal_2.py
python mis_normal_3.py
python mis_unknown_2.py
python mis_unknown_3.py
python mis_known_2.py
python mis_known_3.py
python mis_irrelevant_3.py
```

## Ministral generation settings

```python
gpu_memory_utilization = 0.95
max_model_len = 32000
max_tokens = 32
temperature = 0.0
top_p = 1.0
```

Generation is deterministic under the configured sampling parameters.

## No-image implementation

The four no-image scripts still load and encode the original image but omit it from the `messages` object passed to `llm.chat`.

- `unknown`: prompt describes the task as if a chest X-ray were available.
- `known`: prompt says that the image is not given.

The unnecessary image loading should be removed in a cleanup commit, while retaining the original record's patient and image identifiers.

## Output schema

```json
{
  "patient0001": {
    "sex": "Female",
    "No Finding": 0,
    "images": {
      "image.png": {
        "predictions": [
          {
            "category": "Cardiomegaly",
            "question": "Is cardiomegaly present?",
            "model_answer": "1",
            "expected_answer": 1
          }
        ]
      }
    }
  }
}
```

`model_answer` is a string or `INVALID`.

## Irrelevant-image condition

Both model families currently choose unrelated images using `random.choice`.

Before publication:

- set and record a random seed;
- save the mapping between source CheXpert records and irrelevant images;
- retain source CheXpert patient/image identifiers;
- document sampling with or without replacement.
