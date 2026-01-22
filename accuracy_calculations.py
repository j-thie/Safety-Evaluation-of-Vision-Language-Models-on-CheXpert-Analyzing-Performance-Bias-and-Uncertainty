import csv
from collections import defaultdict
import json

#  Load  JSON file 
json_path = '/path/to/json/file' 

with open(json_path, "r") as f:
    data = json.load(f)


overall_correct = 0
overall_total = 0
invalid_count = 0

category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

for patient in data.values():
    for image in patient["images"].values():
        for pred in image["predictions"]:
            category = pred["category"]
            expected = pred["expected_answer"]
            model_answer = pred["model_answer"]

            # INVALID = force wrong answer
            if model_answer == "INVALID":
                model = 1 - expected
                invalid_count += 1
            else:
                model = int(model_answer)

            overall_total += 1
            category_stats[category]["total"] += 1

            if model == expected:
                overall_correct += 1
                category_stats[category]["correct"] += 1
#  Results 

overall_accuracy = overall_correct / overall_total

print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"INVALID predictions counted as wrong: {invalid_count}\n")

print("Accuracy per category:")
print("-" * 40)

for category, stats in category_stats.items():
    acc = stats["correct"] / stats["total"]
    print(f"{category:30s}  {acc:.4f}  ({acc*100:.2f}%)")

#  Write CSV 

filename = "/Users/add/pathway/med_acc/med_PF_accuracy_results.csv"

with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Header
    writer.writerow(["Category", "Correct", "Total", "Accuracy"])

    # Per-category rows
    for category, stats in category_stats.items():
        accuracy = stats["correct"] / stats["total"]
        writer.writerow([
            category,
            stats["correct"],
            stats["total"],
            round(accuracy, 4)
        ])

    # Overall row
    overall_accuracy = overall_correct / overall_total
    writer.writerow([])
    writer.writerow(["OVERALL", overall_correct, overall_total, round(overall_accuracy, 4)])
    writer.writerow(["INVALID counted as wrong", invalid_count, "", ""])


print(f"CSV file saved as: {filename}")
print(f"Overall Accuracy: {overall_accuracy:.2%}")
