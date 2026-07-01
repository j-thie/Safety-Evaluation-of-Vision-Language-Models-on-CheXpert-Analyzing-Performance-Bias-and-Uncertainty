import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)

# settings

json_path = '/add/path/to/data' # 

projection_questions = ["Frontal_Lateral", "Projection"]

disease_questions = [
    "Enlarged_Cardiomediastinum",
    "Cardiomegaly",
    "Lung_Opacity",
    "Lung_Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural_Effusion",
    "Pleural_Other",
    "Fracture",
    "Support_Devices"
]

#load json file 

with open(json_path, "r") as f:
    data = json.load(f)

# storage

labels = {}

for q in projection_questions + disease_questions:
    labels[q] = {"y_true": [], "y_pred": []}

# parse data 

for patient_id, patient_data in data.items():

    for image_name, image_data in patient_data["images"].items():

        for pred in image_data["predictions"]:

            category = pred["category"]

            if category not in labels:
                continue

            expected = int(pred["expected_answer"])
            model_answer = pred["model_answer"]

            # Handle INVALID → force wrong prediction
            if model_answer == "INVALID":
                predicted = 1 - expected  # flip label
            else:
                predicted = int(model_answer)

            labels[category]["y_true"].append(expected)
            labels[category]["y_pred"].append(predicted)

# evaluation function 

def evaluate(y_true, y_pred, question_type):

    if question_type == "disease":
        avg_type = "binary"
    else:
        avg_type = "macro"

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if question_type == "disease" and cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    else:
        specificity = None

    return accuracy, precision, recall, f1, specificity, cm

# run evaluation 

results = []
confusion_matrices = {}

for question, values in labels.items():

    y_true = np.array(values["y_true"])
    y_pred = np.array(values["y_pred"])

    if len(y_true) == 0:
        continue

    if question in projection_questions:
        q_type = "projection"
    else:
        q_type = "disease"

    accuracy, precision, recall, f1, specificity, cm = evaluate(
        y_true, y_pred, q_type
    )

    row = {
        "Question": question,
        "Type": q_type,
        "Num_Samples": len(y_true),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity
    }

    if cm.shape == (2, 2):
        row.update({
            "TN": cm[0][0],
            "FP": cm[0][1],
            "FN": cm[1][0],
            "TP": cm[1][1],
        })

    results.append(row)
    confusion_matrices[question] = cm

 
save_path = "/change/path/to/where/data/should/be/saved" 

# create directory if it doesn't exist

os.makedirs(save_path, exist_ok=True)

# create dataframe 

df = pd.DataFrame(results)

# save CSV version- change names of the csv, table and confusion matrix files 

csv_file = os.path.join(save_path, "medgemma_prompt_F_trial.csv")
df.to_csv(csv_file, index=False)

# save table

table_file = os.path.join(save_path, "medgemma_prompt_F_table_trial.txt")

with open(table_file, "w") as f:
    f.write(df.to_string(index=False))

# save confusion matrix 

cm_file = os.path.join(save_path, "medgemma_prompt_F_confusion_matrices_trial.txt")

with open(cm_file, "w") as f:
    for question, cm in confusion_matrices.items():
        f.write(f"{question}\n")
        f.write(str(cm))
        f.write("\n\n")

print("Evaluation complete.")
print(f"Saved: {csv_file}")
print(f"Saved: {table_file}")
print(f"Saved: {cm_file}")
