import json
import csv
import os
from collections import defaultdict

MEDGEMMA_FILES = [
            r'/add/path/to/JSON/files/for/all/data.json'
]

MISTRAL_FILES = [
            r'/add/path/to/JSON/files/for/all/data.json'
]


def extract_prompt_name(filepath):
    """Extract prompt label (a, b, c, ...) from filename"""
    filename = os.path.basename(filepath)
    return filename.split("_")[-1].replace(".json", "")


def analyze_file(filepath, model_name):
    with open(filepath, "r") as f:
        data = json.load(f)

    prompt = extract_prompt_name(filepath)

    # Counters
    category_counts = defaultdict(int)
    category_sex_counts = defaultdict(lambda: {"Male": 0, "Female": 0})
    total_invalids = 0
    male_invalids = 0
    female_invalids = 0
    patients_multiple_invalids = 0

    for patient_id, patient_data in data.items():
        sex = patient_data.get("sex", "Unknown")
        patient_invalids = 0

        for image_data in patient_data.get("images", {}).values():
            for pred in image_data.get("predictions", []):
                if str(pred.get("model_answer")).upper() == "INVALID":
                    category = pred.get("category")

                    # Global counts
                    total_invalids += 1
                    category_counts[category] += 1
                    category_sex_counts[category][sex] += 1

                    # Sex totals
                    if sex == "Male":
                        male_invalids += 1
                    elif sex == "Female":
                        female_invalids += 1

                    patient_invalids += 1

        if patient_invalids > 1:
            patients_multiple_invalids += 1

    # Convert to rows
    rows = []
    for category in category_counts:
        rows.append({
            "model": model_name,
            "prompt": prompt,
            "category": category,
            "invalid_count": category_counts[category],
            "male_invalids": category_sex_counts[category]["Male"],
            "female_invalids": category_sex_counts[category]["Female"],
            "total_invalids_prompt": total_invalids,
            "male_total_prompt": male_invalids,
            "female_total_prompt": female_invalids,
            "patients_multiple_invalids": patients_multiple_invalids
        })

    return rows


# Run analysis 
all_rows = []

for file in MEDGEMMA_FILES:
    all_rows.extend(analyze_file(file, "MedGemma"))

for file in MISTRAL_FILES:
    all_rows.extend(analyze_file(file, "Ministral"))


#  Save to CSV 
output_path = "invalid_summary.csv"

with open(output_path, "w", newline="") as csvfile:
    fieldnames = [
        "model",
        "prompt",
        "category",
        "invalid_count",
        "male_invalids",
        "female_invalids",
        "total_invalids_prompt",
        "male_total_prompt",
        "female_total_prompt",
        "patients_multiple_invalids"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved results to {output_path}")