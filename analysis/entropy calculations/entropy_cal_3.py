import json
import math
from collections import Counter
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

# number of valid classes per scenario 
SCENARIO_CLASSES = {
    "Normal": 4,
    "No Image Unknown": 4,
    "No Image Known": 4
}

# helpers
def load_json(path):

    with open(path, "r") as f:
        return json.load(f)


def normalize_answer(ans):
    """
    Valid mappings:

    0 = No
    1 = Yes
    2 = Unknown
    3 = Invalid 


    """

    ans = str(ans).strip()

    if ans == "0":
        return 0

    elif ans == "1":
        return 1

    elif ans == "2":
        return 2

    else:
        return 3


def normalized_entropy(values, num_classes):
    """
    Normalized Shannon entropy.

    Output range:
    0 = complete agreement
    1 = maximal disagreement
    """

    counts = Counter(values)
    total = len(values)

    H = 0

    for c in counts.values():

        p = c / total

        H -= p * math.log2(p)

    H_max = math.log2(num_classes)

    return H / H_max


def extract_answers(prompt_file):
    """
    Returns:
    key = (patient_id, image_name, question_idx)
    value = normalized answer
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

                answer = normalize_answer(
                    pred["model_answer"]
                )

                result[key] = answer

    return result

# main computations 

def scenario_metrics(
    prompt_paths,
    num_classes
):
    """
    Computes mean normalized entropy
    across all cases.
    """

    # Load all prompt results
    prompt_results = [
        extract_answers(path)
        for path in prompt_paths
    ]

    keys = prompt_results[0].keys()

    entropies = []
    invalid_rates = []

    for key in keys:

        answers = [
            prompt_dict[key]
            for prompt_dict in prompt_results
        ]


        # Skip empty cases
        if len(answers) == 0:
            continue

        H = normalized_entropy(
            answers,
            num_classes
        )
        entropies.append(H)

        invalid_count = sum(1 for a in answers if a == 3)
        invalid_rate = invalid_count / len(answers)
        invalid_rates.append(invalid_rate)

    return np.mean(entropies), np.mean(invalid_rates)



results = {}

for scenario_name, models in EXPERIMENTS.items():

    results[scenario_name] = {}

    num_classes = SCENARIO_CLASSES[scenario_name]

    for model_name, prompt_paths in models.items():

        entropy, invalid_rate = scenario_metrics(
            prompt_paths,
            num_classes
        )

        results[scenario_name][model_name] = {
            "entropy": entropy,
            "invalid_rate": invalid_rate
        }


# run all experiments 

results = {}

for scenario_name, models in EXPERIMENTS.items():

    results[scenario_name] = {}

    num_classes = SCENARIO_CLASSES[scenario_name]

    for model_name, prompt_paths in models.items():

        entropy, invalid_rate = scenario_metrics(
            prompt_paths,
            num_classes
        )

        results[scenario_name][model_name] = {
            "entropy": entropy,
            "invalid_rate": invalid_rate
        }


# print results 

for scenario, models in results.items():

    print(f"\n{scenario}")

    for model, metrics in models.items():

        print(
            f"  {model}: "
            f"entropy = {metrics['entropy']:.4f}, "
            f"invalid_rate = {metrics['invalid_rate']:.4f}"
        )