import json
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

#config

DATA_FOLDER = '/add/path/to/JSON/files/for/all/data.json'
DATA_PATH = os.path.join(DATA_FOLDER, "*.json")

files = glob.glob(DATA_PATH)


# load data
files = glob.glob(DATA_PATH)

all_rows = []

for file in files:
    with open(file, "r") as f:
        data = json.load(f)

    filename = os.path.basename(file)
    name_without_ext = os.path.splitext(filename)[0]

    parts = name_without_ext.split("_")

    model = parts[0]
    prompt = "_".join(parts[1:]) 

    for patient_id, patient_data in data.items():
        sex = patient_data["sex"]

        for image_name, image_data in patient_data["images"].items():
            for pred in image_data["predictions"]:

                expected = int(pred["expected_answer"])

                # If INVALID → force wrong prediction
                if pred["model_answer"] == "INVALID":
                    predicted = 1 - expected
                else:
                    predicted = int(pred["model_answer"])

                all_rows.append({
                    "model": model,
                    "prompt": prompt,
                    "sex": sex,
                    "category": pred["category"],
                    "y_true": expected,
                    "y_pred": predicted
                })

df = pd.DataFrame(all_rows)

print("\nTotal samples:", len(df))


# basic metric function

def compute_metrics(group):
    tp = ((group.y_true == 1) & (group.y_pred == 1)).sum()
    tn = ((group.y_true == 0) & (group.y_pred == 0)).sum()
    fp = ((group.y_true == 0) & (group.y_pred == 1)).sum()
    fn = ((group.y_true == 1) & (group.y_pred == 0)).sum()

    return pd.Series({
        "accuracy": (tp + tn) / len(group),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "samples": len(group)
    })


df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)


# overal model x gender

print("\n model x gender matrix")
metrics = (
    df.groupby(["model", "sex"])
    .apply(compute_metrics)
    .reset_index()
)

print(metrics)


# gender gap per model 

print("\n gender gap per model")
pivot = metrics.pivot(index="model", columns="sex")

pivot["accuracy_gap"] = (
    pivot["accuracy"]["Male"] - pivot["accuracy"]["Female"]
)

pivot["fpr_gap"] = (
    pivot["fpr"]["Male"] - pivot["fpr"]["Female"]
)

pivot["fnr_gap"] = (
    pivot["fnr"]["Male"] - pivot["fnr"]["Female"]
)

print(pivot[["accuracy_gap", "fpr_gap", "fnr_gap"]])


# prompt level bias 
print("\n=== PROMPT LEVEL GENDER GAPS ===")

prompt_metrics = (
    df.groupby(["model", "prompt", "sex"])
    .apply(compute_metrics)
    .reset_index()
)

prompt_pivot = prompt_metrics.pivot_table(
    index=["model", "prompt"],
    columns="sex",
    values=["accuracy", "fpr", "fnr"]
)

prompt_pivot["accuracy_gap"] = (
    prompt_pivot["accuracy"]["Male"] -
    prompt_pivot["accuracy"]["Female"]
)

prompt_pivot["fpr_gap"] = (
    prompt_pivot["fpr"]["Male"] -
    prompt_pivot["fpr"]["Female"]
)

prompt_pivot["fnr_gap"] = (
    prompt_pivot["fnr"]["Male"] -
    prompt_pivot["fnr"]["Female"]
)

print(prompt_pivot[["accuracy_gap", "fpr_gap", "fnr_gap"]])


# category level biases 

print("\n\category level gender gaps")

category_metrics = (
    df.groupby(["model", "category", "sex"])
    .apply(compute_metrics)
    .reset_index()
)

category_pivot = category_metrics.pivot_table(
    index=["model", "category"],
    columns="sex",
    values=["accuracy", "fpr", "fnr"]
)

category_pivot["accuracy_gap"] = (
    category_pivot["accuracy"]["Male"] -
    category_pivot["accuracy"]["Female"]
)

category_pivot["fpr_gap"] = (
    category_pivot["fpr"]["Male"] -
    category_pivot["fpr"]["Female"]
)

category_pivot["fnr_gap"] = (
    category_pivot["fnr"]["Male"] -
    category_pivot["fnr"]["Female"]
)

print(category_pivot[["accuracy_gap", "fpr_gap", "fnr_gap"]])



print("\nAnalysis Complete.")
