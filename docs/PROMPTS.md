# Prompt Conditions

The repository contains six prompt variants, A–F, across four response/availability families. The complete wording is retained in [`assets/prompt_conditions.pdf`](assets/prompt_conditions.pdf).

## Prompt tone groups

The six prompt variants are divided into two tone groups:

* **Prompts A–C:** medically toned, using more formal clinical language;
* **Prompts D–F:** naturally toned, using more conversational language.

Within each tone group, the prompts vary by response format and image-availability condition.

## Prompt families

| Family                                           |         Image available | Allowed labels |
| ------------------------------------------------ | ----------------------: | -------------- |
| Binary: normal and no-image-unknown family       | As defined by condition | `0`, `1`       |
| Binary: no-image-known family                    |                      No | `0`, `1`       |
| Three-choice: no-image-known family              |                      No | `0`, `1`, `2`  |
| Three-choice: normal and no-image-unknown family | As defined by condition | `0`, `1`, `2`  |

Label definitions:

* `1`: finding present;
* `0`: finding absent, with uncertainty handling determined by the prompt family;
* `2`: insufficient information.

Every prompt requires the final output format:

```text
Final Answer: X
```

## Keeping code and documentation synchronized

In the uploaded scripts, `PROMPT_NAME` is only used in the output filename. It does not select a prompt template.

For every run, verify that:

1. the embedded text matches the intended A–F template;
2. `PROMPT_NAME` uses the matching name;
3. the output directory identifies the condition and response format.
