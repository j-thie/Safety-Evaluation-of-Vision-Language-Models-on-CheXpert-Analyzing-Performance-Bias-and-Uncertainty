import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


pathways = {
    "MedGemma" : [
        r'/add/path/to/JSON/files/for/all/data.json'
    ],
    "Mistral" : [
        r'/add/path/to/JSON/files/for/all/data.json'
    ]
}

def parse_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    records = []
    for patient_id, patient_data in data.items():
        if 'images' in patient_data:
            for image_id, image_data in patient_data['images'].items():
                if 'predictions' in image_data:
                    for pred in image_data['predictions']:
                        records.append({
                            'patient_id': patient_id,
                            'image_id': image_id,
                            'category': pred.get('category'),
                            'model_answer': pred.get('model_answer'),
                            'expected_answer': pred.get('expected_answer')
                        })
    return pd.DataFrame(records)

def process_answers(df):
    def transform_model_answer(row):
        model_ans = str(row['model_answer']).strip().lower()
        expected_ans = row['expected_answer']
        
        # Rule: Convert 'invalid' to a guaranteed wrong answer
        if model_ans == 'invalid':
            if expected_ans == 1:
                return 0
            elif expected_ans == 0:
                return 1
            elif expected_ans == 2:
                return 0  # If expected is 2, setting model answer to 0 makes it incorrect
            else:
                return -1  # General fallback
        
        # Convert valid numerical strings to integers
        try:
            return int(float(model_ans))
        except ValueError:
            # For any other corrupted/unexpected string response, make it wrong
            return 1 if expected_ans == 0 else 0

    df['cleaned_expected'] = df['expected_answer'].astype(int)
    df['cleaned_model'] = df.apply(transform_model_answer, axis=1)
    return df

def calculate_metrics(df):
    results = []
    categories = df['category'].unique()
    
    for cat in categories:
        cat_df = df[df['category'] == cat]
        y_true = cat_df['cleaned_expected'].values
        y_pred = cat_df['cleaned_model'].values
        
        if len(y_true) == 0:
            continue
            
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Predefine labels [0, 1, 2] to ensure the output array is always structured identically
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
        
        res = {
            'Category': cat,
            'Accuracy': acc,
            'F1_Macro': f1_macro,
            'F1_Class_0': per_class_f1[0],
            'F1_Class_1': per_class_f1[1],
            'F1_Class_2': per_class_f1[2]
        }
        results.append(res)
        
    return pd.DataFrame(results)

# --- Master Pipeline ---
all_results = []

for model_name, files in pathways.items():
    for filepath in files:
        if not os.path.exists(filepath):
            print(f"Skipping missing file: {filepath}")
            continue
            
        # Extract prompt identification (e.g., 'prompt_a') from the filename
        prompt_name = os.path.basename(filepath).split('.')[0].replace('med_normal3_', '').replace('normal3_', '')
        
        # Run step-by-step pipeline
        df = parse_json_file(filepath)
        df = process_answers(df)
        metrics_df = calculate_metrics(df)
        
        # Assign metadata columns
        metrics_df['Model'] = model_name
        metrics_df['Prompt'] = prompt_name
        
        all_results.append(metrics_df)

if all_results:
    # Merge results from all files into a structured master sheet
    master_df = pd.concat(all_results, ignore_index=True)
    cols = ['Model', 'Prompt', 'Category', 'Accuracy', 'F1_Macro', 'F1_Class_0', 'F1_Class_1', 'F1_Class_2']
    master_df = master_df[cols]
    
    # Save detailed data to a CSV for your report charts
    output_csv = 'norm3_f1_acc.csv'
    master_df.to_csv(output_csv, index=False)
    print(f"Success! Detailed metrics saved locally to: {output_csv}\n")
    
    # Print out a clear aggregated view (Averaged across all prompts a-f)
    print("=== SUMMARY: AVERAGE PERFORMANCE PER MODEL AND CATEGORY (ACROSS PROMPTS) ===")
    summary = master_df.groupby(['Model', 'Category'])[['Accuracy', 'F1_Macro', 'F1_Class_0', 'F1_Class_1', 'F1_Class_2']].mean()
    print(summary.to_string())
else:
    print("Execution failed. Please confirm your local absolute JSON paths are accurate.")

import pandas as pd
import io

# Load the raw string data you provided
raw_data = """ add data here """

# If using a generated file from before, uncomment line below:
# df = pd.read_csv('thesis_evaluation_results.csv')
df = pd.read_csv(io.StringIO(raw_data)) # Placeholder for your input block

# Group by Model and Category, then calculate Mean and Standard Deviation across all prompts
grouped = df.groupby(['Model', 'Category']).agg({
    'Accuracy': ['mean', 'std'],
    'F1_Macro': ['mean', 'std']
}).round(3)

# Flatten columns for clean reading
grouped.columns = ['_'.join(col) for col in grouped.columns]
grouped = grouped.reset_index()


grouped['Paper_Accuracy'] = grouped['Accuracy_mean'].astype(str) + " (±" + grouped['Accuracy_std'].fillna(0.0).astype(str) + ")"
grouped['Paper_F1_Macro'] = grouped['F1_Macro_mean'].astype(str) + " (±" + grouped['F1_Macro_std'].fillna(0.0).astype(str) + ")"


paper_table = grouped.pivot(index='Category', columns='Model', values=['Paper_Accuracy', 'Paper_F1_Macro'])

print(paper_table.to_string())