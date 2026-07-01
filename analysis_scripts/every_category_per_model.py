import json
import random
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score

# settings

FILES = [
    r'/add/path/to/JSON/files/for/each/prompt/for/each/model.json'
]

N_BOOT = 10000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# category groups

PROJECTION_CATEGORIES = {"Frontal_Lateral", "Projection"}

def is_projection(category):
    return category in PROJECTION_CATEGORIES

# prediction handling

def parse_prediction(model_answer, expected):

    pred_str = str(model_answer).strip().upper()

    if pred_str == "INVALID":
        return 1 - int(expected)

    return int(pred_str)

# metrics

def accuracy_score_manual(y_true, y_pred):
    return np.mean([t == p for t, p in zip(y_true, y_pred)])

# extract per category 

def extract_per_category_scores(data):

    category_scores = defaultdict(lambda: defaultdict(lambda: ([], [])))

    for pid, pdata in data.items():

        for img in pdata["images"].values():

            for p in img["predictions"]:

                category = p["category"]

                pred = parse_prediction(
                    p["model_answer"],
                    p["expected_answer"]
                )

                y_true, y_pred = category_scores[category][pid]

                y_true.append(int(p["expected_answer"]))
                y_pred.append(pred)

    return category_scores

# bootstrap metrics

def bootstrap_metrics(ps, avg_type="binary"):

    ids = list(ps.keys())

    f1_scores = []
    acc_scores = []

    for _ in range(N_BOOT):

        sample = [random.choice(ids) for _ in ids]

        ys_true = []
        ys_pred = []

        for pid in sample:
            t, p = ps[pid]
            ys_true.extend(t)
            ys_pred.extend(p)

        f1_scores.append(
            f1_score(ys_true, ys_pred, average=avg_type, zero_division=0)
        )

        acc_scores.append(
            accuracy_score_manual(ys_true, ys_pred)
        )

    return {
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "f1_ci": np.percentile(f1_scores, [2.5, 97.5]),
        "acc_mean": np.mean(acc_scores),
        "acc_std": np.std(acc_scores),
        "acc_ci": np.percentile(acc_scores, [2.5, 97.5])
    }

# parse file name 

def parse_name(filepath):

    stem = Path(filepath).stem
    model, _, prompt = stem.split("_")

    return model, prompt

# aggregate all results 

def aggregate_model_results():

    model_category_results = defaultdict(lambda: defaultdict(list))

    for f in FILES:

        with open(f) as fh:
            data = json.load(fh)

        model, prompt = parse_name(f)

        category_scores = extract_per_category_scores(data)

        for category, ps in category_scores.items():

            if len(ps) < 5:
                continue

            avg_type = "macro" if is_projection(category) else "binary"

            metrics = bootstrap_metrics(ps, avg_type)

            model_category_results[model][category].append(metrics)

    return model_category_results


# summarize across prompts 

def summarize_model(model_results):

    final_rows = []

    for category, runs in model_results.items():

        f1_means = [r["f1_mean"] for r in runs]
        acc_means = [r["acc_mean"] for r in runs]

        f1_stds = [r["f1_std"] for r in runs]
        acc_stds = [r["acc_std"] for r in runs]

        f1_ci_low = np.mean([r["f1_ci"][0] for r in runs])
        f1_ci_high = np.mean([r["f1_ci"][1] for r in runs])

        acc_ci_low = np.mean([r["acc_ci"][0] for r in runs])
        acc_ci_high = np.mean([r["acc_ci"][1] for r in runs])

        final_rows.append({
            "category": category,
            "f1_mean": np.mean(f1_means),
            "f1_std": np.mean(f1_stds),
            "f1_ci": (f1_ci_low, f1_ci_high),
            "acc_mean": np.mean(acc_means),
            "acc_std": np.mean(acc_stds),
            "acc_ci": (acc_ci_low, acc_ci_high),
        })

    return final_rows

# formal output

def format_metric(mean, std, ci):

    return f"{mean:.3f} ± {std:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]"

# write final tables 

def write_final_tables():

    results = aggregate_model_results()

    for model, model_results in results.items():

        rows = summarize_model(model_results)

        out_file = f"final_table_{model}.csv"

        with open(out_file, "w", newline="") as f:

            writer = csv.writer(f, delimiter="\t")

            writer.writerow([
                "Finding / Label",
                "Mean F1 ± SD\nCI",
                "Mean Acc ± SD\nCI"
            ])

            for r in rows:

                writer.writerow([
                    r["category"],
                    format_metric(r["f1_mean"], r["f1_std"], r["f1_ci"]),
                    format_metric(r["acc_mean"], r["acc_std"], r["acc_ci"]),
                ])

        print("Saved:", out_file)

# run

write_final_tables()