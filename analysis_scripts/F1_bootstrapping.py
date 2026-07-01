import json
import random
import numpy as np
import csv
import re
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from sklearn.metrics import f1_score

# settings

FILES = [
    r'/add/path/to/JSON/files/for/all/data.json'
]

N_BOOT = 10000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# category groups 
PROJECTION_CATEGORIES = {"Frontal_Lateral", "Projection"}


def is_projection(category):
    return category in PROJECTION_CATEGORIES


def is_disease(category):
    return category not in PROJECTION_CATEGORIES

zero_division=0

# core helpers 

def extract_patient_scores(data, group="all"):
    """
    Returns:
        { patient_id : (y_true_list, y_pred_list) }
    """
    patient_scores = {}

    for pid, pdata in data.items():
        y_true = []
        y_pred = []

        for img in pdata["images"].values():
            for p in img["predictions"]:
                category = p["category"]

                if group == "disease" and not is_disease(category):
                    continue
                if group == "projection" and not is_projection(category):
                    continue

                pred_str = str(p["model_answer"]).strip().upper()

                if pred_str == "INVALID":
                    pred = 1 - int(p["expected_answer"])
                else:
                    pred = int(pred_str)

                y_pred.append(pred)
                y_true.append(int(p["expected_answer"]))

        if y_true:
            patient_scores[pid] = (y_true, y_pred)

    return patient_scores


# metric functions (F1 based)

def f1_from_scores(ps, group):
    ys_true, ys_pred = [], []

    for t, p in ps.values():
        ys_true.extend(t)
        ys_pred.extend(p)

    avg = "binary" if group == "disease" else "macro"
    return f1_score(ys_true, ys_pred, average=avg)


def bootstrap_f1(ps, group):
    ids = list(ps.keys())
    means = []
    avg = "binary" if group == "disease" else "macro"

    for _ in range(N_BOOT):
        sample = [random.choice(ids) for _ in ids]

        ys_true, ys_pred = [], []

        for pid in sample:
            t, p = ps[pid]
            ys_true.extend(t)
            ys_pred.extend(p)

        means.append(f1_score(ys_true, ys_pred, average=avg))

    return np.mean(means), np.percentile(means, [2.5, 97.5])


def paired_bootstrap_diff(psA, psB, group):
    ids = sorted(set(psA) & set(psB))

    if not ids:
        raise ValueError("No common patients between paired files")
    diffs = []
    avg = "binary" if group == "disease" else "macro"

    for _ in range(N_BOOT):
        sample = [random.choice(ids) for _ in ids]

        tA, pA, tB, pB = [], [], [], []

        for pid in sample:
            a_t, a_p = psA[pid]
            b_t, b_p = psB[pid]

            tA.extend(a_t)
            pA.extend(a_p)
            tB.extend(b_t)
            pB.extend(b_p)

        f1A = f1_score(tA, pA, average=avg)
        f1B = f1_score(tB, pB, average=avg)

        diffs.append(f1A - f1B)

    mean_diff = np.mean(diffs)
    ci = np.percentile(diffs, [2.5, 97.5])
    significant = not (ci[0] <= 0 <= ci[1])

    return mean_diff, ci, significant


def parse_name(filepath):
    stem = Path(filepath).stem

    model = (
        "MedGemma"
        if stem.lower().startswith(("mg_", "medgemma"))
        else "Ministral"
    )

    match = re.search(r"prompt_?([a-f])", stem, re.IGNORECASE)
    prompt = f"prompt_{match.group(1).lower()}" if match else "unknown"

    return model, prompt


# main analysis

def run_analysis(group_name):

    print(f"\nRunning analysis for: {group_name.upper()}")

    OUT_PER_FILE = f"results_per_file_{group_name}.csv"
    OUT_MODEL_SUMMARY = f"results_model_summary_{group_name}.csv"
    OUT_PAIRWISE = f"results_pairwise_significance_{group_name}.csv"

    all_patient_scores = {}
    per_file_rows = []

    # per file analysis

    for f in FILES:
        with open(f) as fh:
            data = json.load(fh)

        ps = extract_patient_scores(data, group=group_name)
        all_patient_scores[f] = ps

        overall = f1_from_scores(ps, group_name)
        boot_mean, ci = bootstrap_f1(ps, group_name)

        model, prompt = parse_name(f)

        per_file_rows.append([
            Path(f).stem,
            model,
            prompt,
            overall,
            boot_mean,
            ci[0],
            ci[1]
        ])

    # save per file SVC

    with open(OUT_PER_FILE, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "file",
            "model",
            "prompt",
            "f1",
            "bootstrap_mean",
            "ci_low",
            "ci_high"
        ])
        writer.writerows(per_file_rows)

    # model summary 

    model_groups = defaultdict(list)

    for row in per_file_rows:
        model_groups[row[1]].append(row[4])

    summary_rows = []

    for model, vals in model_groups.items():
        summary_rows.append([
            model,
            np.mean(vals),
            np.std(vals)
        ])

    with open(OUT_MODEL_SUMMARY, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["model", "mean_f1", "std_f1"])
        writer.writerows(summary_rows)

    # pairwise significance  

    pair_rows = []

    for fA, fB in combinations(FILES, 2):
        psA = all_patient_scores[fA]
        psB = all_patient_scores[fB]

        diff, ci, sig = paired_bootstrap_diff(psA, psB, group_name)

        pair_rows.append([
            Path(fA).stem,
            Path(fB).stem,
            diff,
            ci[0],
            ci[1],
            "YES" if sig else "NO"
        ])

    with open(OUT_PAIRWISE, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "file_A",
            "file_B",
            "mean_diff(A-B)",
            "ci_low",
            "ci_high",
            "significant_95"
        ])
        writer.writerows(pair_rows)

    print("Saved:")
    print(OUT_PER_FILE)
    print(OUT_MODEL_SUMMARY)
    print(OUT_PAIRWISE)


# run both groups 

run_analysis("disease")
run_analysis("projection")
