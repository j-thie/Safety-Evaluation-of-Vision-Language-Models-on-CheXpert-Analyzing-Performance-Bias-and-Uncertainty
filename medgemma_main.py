import json
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


#  data extraction from json file

def flatten_questions(entry):
    question_list = []
    ignore_keys = ["image_path", "sex", "No Finding", "patient_id", "study_id", "view"]

    for key, value in entry.items():
        if key in ignore_keys:
            continue
        if isinstance(value, list):
            for q in value:
                question_list.append({
                    "finding": key,
                    "question": q["question"],
                    "answer": q["answer"]
                })
    return question_list


#  binary responses 

def extract_binary_answer(text):
    text = text.strip()
    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1]
    if "1" in text:
        return "1"
    if "0" in text:
        return "0"
    return "INVALID"


#  extract patient id 

def extract_patient_id(image_path):
    for part in image_path.split("/"):
        if part.startswith("patient"):
            return part
    return "UNKNOWN_PATIENT"


#  extract image path 

def extract_image_id(image_path):
    # e.g. view1_frontal.jpg or view2_lateral.jpg
    return os.path.basename(image_path)


# ---------------------------------------------------------

PROMPT_NAME = "prompt_f" #change this for the different prompts 

def main():


    #  Load JSON file
    json_path = "/path/to/json/file/chexpert_qa_long.json"
    with open(json_path, "r") as f:
        entries = json.load(f)

    #  Load Med-Gemma model from local path
    model_dir = "/home/ul/ul_student/ul_sau95/medgemma/medgemma-4b-it"
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Model device:", next(model.parameters()).device)

    # patient_id -> image_id -> list of question results
    all_results = {}

    # Loop over all entries (supports multiple images per patient)
    for entry in entries:

        image_path = entry["image_path"]

        if not os.path.exists(image_path):
            print(f"WARNING: Missing image {image_path}")
            continue

        patient_id = extract_patient_id(image_path)
        image_id = extract_image_id(image_path)

        image = Image.open(image_path).convert("RGB")
        questions = flatten_questions(entry)

        if patient_id not in all_results:
            all_results[patient_id] = {
                "sex": entry.get("sex", "UNKNOWN"), 
                "No Finding": entry.get("No Finding", None), 
                "images":{}
            }

        if image_id not in all_results[patient_id]["images"]:
            all_results[patient_id]["images"][image_id] = {
                "predictions": []
            }


        for q in questions:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                f"Prompt: {PROMPT_NAME}\n\n" # replace lines 120 - 128 with different prompts 
                                "This is a medical image classification task.\n"
                                "Evaluate whether the listed finding is visible in the image.\n\n"
                                "Return ONLY the following format:\n"
                                "Final Answer: X\n\n"
                                "Where:\n"
                                "1 = Yes (finding present)\n"
                                "0 = No  (finding absent or uncertain)\n\n"
                                "The image is a chest X-ray.\n"
                                f"Finding: {q['question']}"

                            ),
                        },
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = processor(
                text=text,
                images=[image],
                return_tensors="pt"
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )

            generated_ids = output[0][inputs["input_ids"].shape[-1]:]
            answer_text = processor.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()

            answer = extract_binary_answer(answer_text)

            all_results[patient_id]["images"][image_id]["predictions"].append({
                "category": q["finding"],
                "question": q["question"],
                "model_answer": answer,
                "expected_answer": q["answer"]
            })



    #  printed results 
    for patient_id, pdata in all_results.items():
        print(f"PATIENT: {patient_id}")
        print(f"Sex: {pdata['sex']}")
        print(f"No Finding: {pdata['No Finding']}")

        for image_id, results in pdata["images"].items():
            print(f"\n--- IMAGE: {image_id} ---\n")

            for r in results["predictions"]:
                print(f"[{r['category']}]")
                print(f"Q: {r['question']}")
                print(f"Model answer:          {r['model_answer']}")
                print(f"Expected answer:       {r['expected_answer']}")
                print()

        print("\n")
 

    # save results
    output_dir = "/pathway/where/to/save/results"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"MedGemma_{PROMPT_NAME}.json")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()




