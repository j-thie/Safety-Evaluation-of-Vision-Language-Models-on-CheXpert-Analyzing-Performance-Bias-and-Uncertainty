import json
import numpy as np
import csv


MED_FILES = [
    r'/add/path/to/JSON/files/for/medGemma_data/for/each/prompt.json'

]

MIS_FILES = [
    r'/add/path/to/JSON/files/for/mistral_data/for/each/prompt.json'
]

# --------------------------------------------------------------

def extract_accuracy_by_category(json_path):

    with open(json_path) as f:
        data = json.load(f)

    category_dict = {}

    for patient in data.values():

        for img in patient["images"].values():

            for pred in img["predictions"]:

                category = pred["category"]
                expected = pred["expected_answer"]

                # Treat INVALID as always wrong
                if pred["model_answer"] == "INVALID":
                    model = 1 - expected
                else:
                    model = int(pred["model_answer"])

                correct = int(model == expected)

                if category not in category_dict:
                    category_dict[category] = []

                category_dict[category].append(correct)

    return category_dict

# --------------------------------------------------------------

def pool_by_category(files):

    pooled = {}

    for f in files:

        file_data = extract_accuracy_by_category(f)

        for category, values in file_data.items():

            if category not in pooled:
                pooled[category] = []

            pooled[category].extend(values)

    return pooled

# --------------------------------------------------------------

def bootstrap_ci(data, n_boot=10000):

    data = np.array(data)
    n = len(data)
    means = np.empty(n_boot)

    for i in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)

    return np.percentile(means, [2.5, 97.5])

# --------------------------------------------------------------

def bootstrap_diff_ci(data1, data2, n_boot=10000):

    n1 = len(data1)
    n2 = len(data2)
    diffs = np.empty(n_boot)

    for i in range(n_boot):
        s1 = np.random.choice(data1, size=n1, replace=True)
        s2 = np.random.choice(data2, size=n2, replace=True)
        diffs[i] = np.mean(s1) - np.mean(s2)

    return np.percentile(diffs, [2.5, 97.5])

# --------------------------------------------------------------

print("\nPooling data by category...")

MED = pool_by_category(MED_FILES)
MIS_MED = pool_by_category(MIS_FILES)

print("\nBootstrapping per category...")

results = []

for category in MED.keys():

    if category not in NONMED:
        continue

    med_vals = np.array(MED[category])
    nonmed_vals = np.array(NONMED[category])

    ci_med = bootstrap_ci(med_vals)
    ci_nonmed = bootstrap_ci(nonmed_vals)
    ci_diff = bootstrap_diff_ci(med_vals, nonmed_vals)

    results.append([
        category,
        len(med_vals),
        np.mean(med_vals),
        ci_med[0],
        ci_med[1],
        len(nonmed_vals),
        np.mean(nonmed_vals),
        ci_nonmed[0],
        ci_nonmed[1],
        np.mean(med_vals) - np.mean(nonmed_vals),
        ci_diff[0],
        ci_diff[1]
    ])

# --------------------------------------------------------------
# Write Google Sheets Ready CSV

with open("per_category_bootstrap_results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Category",
        "Med_N",
        "Med_Mean",
        "Med_CI_Lower",
        "Med_CI_Upper",
        "NonMed_N",
        "NonMed_Mean",
        "NonMed_CI_Lower",
        "NonMed_CI_Upper",
        "Diff_Med-NonMed",
        "Diff_CI_Lower",
        "Diff_CI_Upper"
    ])

    writer.writerows(results)

print("\nSaved results to: per_category_bootstrap_results.csv\n")
