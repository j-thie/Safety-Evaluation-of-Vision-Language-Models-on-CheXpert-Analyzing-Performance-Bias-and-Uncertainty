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


def compute_metrics(rows, view_type):
    subset = [r for r in rows if r["view_type"] == view_type]

    if not subset:
        return {
            "accuracy": 0,
            "f1": 0,
            "total": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0
        }

    tp = fp = fn = tn = 0

    for r in subset:
        y_true = r["expected_answer"]
        y_pred = r["model_answer"]

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1

    accuracy = (tp + tn) / len(subset)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "total": len(subset),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


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

                    # handle invalid
                    if raw_model_answer not in ["0", "1"]:
                        is_invalid = True
                        model_answer = 1 - expected_answer  # flip
                        correctness = "wrong"
                    else:
                        is_invalid = False
                        model_answer = int(raw_model_answer)
                        correctness = "correct" if model_answer == expected_answer else "wrong"

                    # determine view type
                    view_lower = view_name.lower()
                    if "frontal" in view_lower:
                        view_type = "frontal"
                    elif "lateral" in view_lower:
                        view_type = "lateral"
                    else:
                        view_type = "unknown"

                    rows.append({
                        "model": model_name,
                        "prompt": prompt,
                        "patient_id": patient_id,
                        "sex": sex,
                        "view": view_name,
                        "view_type": view_type,
                        "question": pred.get("question"),
                        "model_answer": model_answer,
                        "expected_answer": expected_answer,
                        "is_invalid": is_invalid,
                        "correctness": correctness
                    })

    return rows


# collect data
all_rows = []
all_rows.extend(process_files(MEDGEMMA_FILES, "MedGemma"))
all_rows.extend(process_files(MISTRAL_FILES, "Mistral"))


# write main CSV
output_file = "frontal_lateral_analysis.csv"

with open(output_file, "w", newline="") as csvfile:
    fieldnames = [
        "model", "prompt", "patient_id", "sex",
        "view", "view_type",
        "question",
        "model_answer", "expected_answer",
        "is_invalid", "correctness"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Saved to {output_file}")


# write metrics CVS
metrics_file = "frontal_lateral_metrics_detailed.csv"

with open(metrics_file, "w", newline="") as csvfile:
    fieldnames = [
        "model", "prompt", "view_type",
        "samples", "accuracy", "f1",
        "tp", "fp", "fn", "tn"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for model in ["MedGemma", "Mistral"]:
        model_rows = [r for r in all_rows if r["model"] == model]
        prompts = sorted(set(r["prompt"] for r in model_rows))

        for prompt in prompts:
            prompt_rows = [r for r in model_rows if r["prompt"] == prompt]

            for vt in ["frontal", "lateral"]:
                m = compute_metrics(prompt_rows, vt)

                writer.writerow({
                    "model": model,
                    "prompt": prompt,
                    "view_type": vt,
                    "samples": m["total"],
                    "accuracy": round(m["accuracy"], 4),
                    "f1": round(m["f1"], 4),
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "tn": m["tn"]
                })

print(f"Saved to {metrics_file}")


# print metrics 
for model in ["MedGemma", "Mistral"]:
    print(f"\nModel: {model}")

    model_rows = [r for r in all_rows if r["model"] == model]
    prompts = sorted(set(r["prompt"] for r in model_rows))

    for prompt in prompts:
        print(f"\n  Prompt: {prompt}")

        prompt_rows = [r for r in model_rows if r["prompt"] == prompt]

        for vt in ["frontal", "lateral"]:
            m = compute_metrics(prompt_rows, vt)

            print(f"    {vt.capitalize()}:")
            print(f"      Samples: {m['total']}")
            print(f"      Accuracy: {m['accuracy']:.4f}")
            print(f"      F1: {m['f1']:.4f}")
            print(f"      TP:{m['tp']} FP:{m['fp']} FN:{m['fn']} TN:{m['tn']}")