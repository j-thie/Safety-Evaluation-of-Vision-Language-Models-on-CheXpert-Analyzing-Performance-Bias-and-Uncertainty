import json
import numpy as np
import os

EXPERIMENTS = {
    "Normal": {
        "MedGemma": [
            r'/add/path/to/JSON/files/for/all/data.json'
        ],
        "Mistral": [
            r'/add/path/to/JSON/files/for/all/data.json'
        ]
    },

    "No Image Unknown": {
        "MedGemma": [
            r'/add/path/to/JSON/files/for/all/data.json'
        ],
        "Mistral": [
            r'/add/path/to/JSON/files/for/all/data.json'
        ]
    },

    "No Image Known": {
        "MedGemma": [
            r'/add/path/to/JSON/files/for/all/data.json'
        ],
        "Mistral": [
            r'/add/path/to/JSON/files/for/all/data.json'
        ]
    }
}

# helpers
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_answer(ans):
    """
    Map answers:
    0 = No
    1 = Yes
    2 = Invalid / other
    """
    ans = str(ans).strip()

    if ans == "0":
        return 0
    elif ans == "1":
        return 1
    else:
        return 2


def extract_answers(prompt_file):
    """
    Returns:
    key = (patient_id, image_name, question_idx)
    value = answer
    """
    data = load_json(prompt_file)

    result = {}

    for patient_id, patient_data in data.items():

        for image_name, image_data in patient_data["images"].items():

            for q_idx, pred in enumerate(
                image_data["predictions"]
            ):

                key = (
                    patient_id,
                    image_name,
                    q_idx
                )

                result[key] = normalize_answer(
                    pred["model_answer"]
                )

    return result


# SD metric
def std_uncertainty(prompt_paths):
    """
    Compute mean SD across all cases
    using 6 prompt answers.
    """

    # Load all 6 prompt files
    prompt_results = [
        extract_answers(path)
        for path in prompt_paths
    ]

    keys = prompt_results[0].keys()

    stds = []

    for key in keys:

        answers = [
            prompt_dict[key]
            for prompt_dict in prompt_results
        ]

        sd = np.std(answers)

        stds.append(sd)

    return np.mean(stds)


for scenario, models in EXPERIMENTS.items():
    for model, files in models.items():
        for file in files:
            if not os.path.exists(file):
                print("Missing:", file)


# run all experiments 
results = {}

for scenario_name, models in EXPERIMENTS.items():

    results[scenario_name] = {}

    for model_name, prompt_paths in models.items():

        uncertainty_score = std_uncertainty(
            prompt_paths
        )

        results[scenario_name][
            model_name
        ] = uncertainty_score


# print results 
for scenario, models in results.items():

    print(f"\n{scenario}")

    for model, score in models.items():

        print(
            f"  {model}: {score:.4f}"
        )