import json
from collections import defaultdict
import re

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


def extract_prompt_name(path):
    match = re.search(r'prompt_([a-f])', path)
    return match.group(1) if match else "unknown"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# Added "total": 0 to track all prediction entries
results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"2": 0, "INVALID": 0, "total": 0})))

for experiment, models in EXPERIMENTS.items():
    for model, paths in models.items():

        model_name = model.lower()

        for path in paths:
            prompt = extract_prompt_name(path)
            data = load_json(path)

            for patient_id, patient_data in data.items():
                images = patient_data.get("images", {})

                for img_name, img_data in images.items():
                    predictions = img_data.get("predictions", [])

                    for pred in predictions:
                        answer = pred.get("model_answer")
                        
                        # Increment total entries for this prompt configuration
                        results[experiment][model_name][prompt]["total"] += 1

                        if answer == "2":
                            results[experiment][model_name][prompt]["2"] += 1
                        elif answer == "INVALID":
                            results[experiment][model_name][prompt]["INVALID"] += 1


# Print structured summary grouped by Experiment
for experiment, models in results.items():
    print(f"EXPERIMENT: {experiment}")
    
    
    for model, prompts in models.items():
        print(f"\n  MODEL: {model}")
        for prompt, counts in sorted(prompts.items()):  # Sorted so prompts read a, b, c...
            total = counts["total"]
            twos = counts["2"]
            invalid = counts["INVALID"]
            
            # Safely calculate percentage to prevent ZeroDivisionError
            percentage_2s = (twos / total * 100) if total > 0 else 0.0
            
            print(f"    Prompt {prompt}: 2s = {twos} ({percentage_2s:.2f}%), INVALID = {invalid}, Total Entries = {total}")