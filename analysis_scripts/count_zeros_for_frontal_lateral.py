import json
from collections import defaultdict

FILES = [
    r'/add/path/to/JSON/files/for/all/data.json'
]

# overall counter
overall_model_0 = 0
overall_expected_0 = 0
overall_invalid = 0

# per prompt counter
per_prompt_stats = defaultdict(lambda: {
    "model_answer_0": 0,
    "expected_answer_0": 0,
    "invalid_answers": 0,
    "total_frontal_lateral_questions": 0
})

# loop through files
for file_path in FILES:

    # prompt name (a, b, c...)
    prompt_name = file_path.split("_")[-1].replace(".json", "")

    with open(file_path, "r") as f:
        data = json.load(f)

    for patient_id, patient_data in data.items():

        images = patient_data.get("images", {})

        for image_name, image_data in images.items():

            predictions = image_data.get("predictions", [])

            for pred in predictions:

                if pred.get("category") == "Frontal_Lateral":

                    per_prompt_stats[prompt_name]["total_frontal_lateral_questions"] += 1

                    model_answer_raw = str(pred.get("model_answer")).strip()
                    expected_answer = int(pred.get("expected_answer"))

                    # count expected 0
                    if expected_answer == 0:
                        overall_expected_0 += 1
                        per_prompt_stats[prompt_name]["expected_answer_0"] += 1

                    # valid numeric answers
                    if model_answer_raw in ["0", "1"]:

                        model_answer = int(model_answer_raw)

                        if model_answer == 0:
                            overall_model_0 += 1
                            per_prompt_stats[prompt_name]["model_answer_0"] += 1

                    # invalid answers
                    else:
                        overall_invalid += 1
                        per_prompt_stats[prompt_name]["invalid_answers"] += 1

# print results 

print("===== OVERALL =====")
print(f"Model answered 0: {overall_model_0}")
print(f"Expected answer was 0: {overall_expected_0}")
print(f"INVALID answers: {overall_invalid}")

print("\n===== PER PROMPT =====")

for prompt, stats in per_prompt_stats.items():

    print(f"\nPrompt: {prompt}")
    print(f"Total Frontal_Lateral questions: {stats['total_frontal_lateral_questions']}")
    print(f"Model answered 0: {stats['model_answer_0']}")
    print(f"Expected answer was 0: {stats['expected_answer_0']}")
    print(f"INVALID answers: {stats['invalid_answers']}")