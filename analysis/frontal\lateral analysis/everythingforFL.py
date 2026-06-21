import json
import csv
import os

MEDGEMMA_FILES = [
    r'/add/path/to/JSON/files/for/all/data.json'
]

MISTRAL_FILES = [
    r'/add/path/to/JSON/files/for/all/data.json'
]


def extract_prompt_name(filepath):
    base = os.path.basename(filepath)
    return base.split('_')[-1].replace('.json', '')


def process_files(file_list, model_name):
    rows = []

    for filepath in file_list:
        prompt = extract_prompt_name(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        for patient_id, patient_data in data.items():
            sex = patient_data.get("sex", "Unknown")

            for view_name, view_data in patient_data.get("images", {}).items():
                predictions = view_data.get("predictions", [])

                for pred in predictions:
                    if pred.get("category") != "Frontal_Lateral":
                        continue

                    expected_answer = int(pred.get("expected_answer", 0))
                    raw_model_answer = str(pred.get("model_answer")).strip()

                    # ---- HANDLE INVALID ----
                    if raw_model_answer not in ["0", "1"]:
                        is_invalid = True
                        model_answer = 1 - expected_answer  # flip
                        correctness = "wrong"
                    else:
                        is_invalid = False
                        model_answer = int(raw_model_answer)
                        correctness = "correct" if model_answer == expected_answer else "wrong"

                    rows.append({
                        "model": model_name,
                        "prompt": prompt,
                        "patient_id": patient_id,
                        "sex": sex,
                        "view": view_name,
                        "question": pred.get("question"),
                        "model_answer": model_answer,
                        "expected_answer": expected_answer,
                        "is_invalid": is_invalid,
                        "correctness": correctness
                    })

    return rows


# Collect all data
all_rows = []
all_rows.extend(process_files(MEDGEMMA_FILES, "MedGemma"))
all_rows.extend(process_files(MISTRAL_FILES, "Mistral"))


# Write CSV
output_file = "frontal_lateral_analysis.csv"

with open(output_file, "w", newline="") as csvfile:
    fieldnames = [
        "model", "prompt", "patient_id", "sex",
        "view", "question",
        "model_answer", "expected_answer",
        "is_invalid", "correctness"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved to {output_file}")
