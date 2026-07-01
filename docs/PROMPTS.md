# Prompt Conditions

The repository contains six prompt variants, A-F, across four response/availability families. The complete wording is retained in [`assets/prompt_conditions.pdf`](assets/prompt_conditions.pdf).

## Prompt families

| Family | Image available | Allowed labels |
|---|---:|---|
| Binary: normal and no-image-unknown family | As defined by condition | `0`, `1` |
| Binary: no-image-known family | No | `0`, `1` |
| Three-choice: no-image-known family | No | `0`, `1`, `2` |
| Three-choice: normal and no-image-unknown family | As defined by condition | `0`, `1`, `2` |

Label definitions:

- `1`: finding present;
- `0`: finding absent, with uncertainty handling determined by the prompt family;
- `2`: insufficient information.

Every prompt requires the final output format:

```text
Final Answer: X
```

## Variants A-F

The variants alter framing and role instructions:

- **A:** compact image type and decision-rule format;
- **B:** direct classification question;
- **C:** precise image-classifier role;
- **D:** radiologist role with conservative interpretation;
- **E:** image-classifier/radiology framing;
- **F:** generic medical-image classification framing.

## Keeping code and documentation synchronized

In the uploaded scripts, `PROMPT_NAME` is only used in the output filename. It does not select a prompt template.

For every run, verify that:

1. the embedded text matches the intended A-F template;
2. `PROMPT_NAME` uses the matching name;
3. the parser accepts the labels advertised by the prompt; and
4. the output directory identifies the condition and response format.

The recommended refactor is a shared `prompts.py` file:

```python
PROMPTS = {
    "binary_normal": {
        "prompt_a": "...",
        "prompt_b": "...",
    },
    "three_choice_no_image": {
        "prompt_a": "...",
        "prompt_b": "...",
    },
}
```

Do not claim this refactor is implemented until all inference scripts import and use it.
