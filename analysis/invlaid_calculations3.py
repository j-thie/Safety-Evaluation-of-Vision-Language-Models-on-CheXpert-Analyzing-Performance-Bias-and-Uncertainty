import json
import numpy as np
from collections import defaultdict

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

#  helpers
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_answer(x):
    """
    Keep answers exactly as categories:
    '0', '1', 'INVALID'
    """
    return str(x).strip().upper()


def extract_predictions(data):
    preds = {}

    for patient_id, patient_data in data.items():

        if "images" not in patient_data:
            continue

        for image_name, image_data in patient_data["images"].items():

            for pred in image_data["predictions"]:

                key = (
                    patient_id,
                    image_name,
                    pred["category"]
                )

                answer = parse_answer(pred["model_answer"])
                expected = str(pred["expected_answer"])

                preds[key] = (answer, expected)

    return preds


def variability_score(answers):
    """
    Variability across outputs:
    {'0','1', '2', 'INVALID'}

    0 = all prompts gave same output
    higher = more disagreement
    """
    counts = defaultdict(int)

    for a in answers:
        counts[a] += 1

    probs = np.array(list(counts.values())) / len(answers)

    # consistency = largest class proportion
    # variability = 1 - consistency
    return 1 - np.max(probs)




# main
results = []

for scenario, models in EXPERIMENTS.items():

    for model_name, prompt_files in models.items():

        prompt_data = []

        for path in prompt_files:
            data = load_json(path)
            prompt_data.append(extract_predictions(data))

        # all cases existing in all prompts
        common_keys = set(prompt_data[0].keys())

        for pd in prompt_data[1:]:
            common_keys &= set(pd.keys())

        case_scores = []

        for key in common_keys:

            answers = []
            expected = None

            for pd in prompt_data:
                answer, exp = pd[key]

                answers.append(answer)

                if expected is None:
                    expected = exp

            # keep only INVALID cases:
            # at least one prompt is wrong
            invalid = any(a != expected for a in answers)

            if not invalid:
                continue

            score = variability_score(answers)
            case_scores.append(score)

        mean_var = np.mean(case_scores) if case_scores else np.nan

        results.append({
            "scenario": scenario,
            "model": model_name,
            "mean_invalid_variability": mean_var,
            "n_invalid_cases": len(case_scores)
        })


# print
for r in results:
    print(
        f"{r['scenario']:20s} | "
        f"{r['model']:10s} | "
        f"mean variability= {r['mean_invalid_variability']:.4f} | "
        f"N= {r['n_invalid_cases']}"
    )