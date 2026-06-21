import json
import math
from collections import Counter
import numpy as np

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
    Map answer into:
    0 = No
    1 = Yes
    2 = Invalid / anything else
    """
    ans = str(ans).strip()

    if ans == "0":
        return 0
    elif ans == "1":
        return 1
    else:
        return 2


def entropy(values):
    """
    Shannon entropy of categorical answers.
    """
    counts = Counter(values)
    total = len(values)

    H = 0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)

    return H / math.log2(3)


def extract_answers(prompt_file):
    """
    Returns dict:
    key = (patient_id, image_name, question_idx)
    value = answer
    """
    data = load_json(prompt_file)

    result = {}

    for patient_id, patient_data in data.items():

        for image_name, image_data in patient_data["images"].items():

            for q_idx, pred in enumerate(image_data["predictions"]):

                key = (patient_id, image_name, q_idx)

                result[key] = normalize_answer(
                    pred["model_answer"]
                )

    return result


# main computation
def scenario_uncertainty(prompt_paths):
    """
    prompt_paths = 6 prompt json files
    """

    # load all 6 prompts
    prompt_results = [
        extract_answers(path)
        for path in prompt_paths
    ]

    # all keys (cases)
    keys = prompt_results[0].keys()

    entropies = []

    for key in keys:

        answers = [
            prompt_dict[key]
            for prompt_dict in prompt_results
        ]

        H = entropy(answers)
        entropies.append(H)

    return np.mean(entropies)


# run all experiments 
results = {}

for scenario_name, models in EXPERIMENTS.items():

    results[scenario_name] = {}

    for model_name, prompt_paths in models.items():

        U = scenario_uncertainty(prompt_paths)

        results[scenario_name][model_name] = U


# print
for scenario, models in results.items():

    print(f"\n{scenario}")

    for model, score in models.items():

        print(f"  {model}: {score:.4f}")