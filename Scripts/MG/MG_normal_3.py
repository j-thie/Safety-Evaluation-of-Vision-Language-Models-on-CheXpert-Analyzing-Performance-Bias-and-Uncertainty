import json
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import re


# data extraction from JSON file 

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


# extract answer 0,1,2

def extract_answer(text):
    print(f"Raw model output: '{text}'")

    text = text.strip()

    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1].strip()

    # Find first standalone 0/1/2 anywhere in remaining text
    match = re.search(r'\b([012])\b', text)

    if match:
        return match.group(1)

    print("Could not extract valid answer")
    return "INVALID"



# extract patient ID from image path 

def extract_patient_id(image_path):
    for part in image_path.split("/"):
        if part.startswith("patient"):
            return part
    return "UNKNOWN_PATIENT"


#  extract image ID 

def extract_image_id(image_path):
    return os.path.basename(image_path)


# main script 

PROMPT_NAME = "prompt_b" # change this as necessary for the appropriate prompt option 

def main():

    print("[INFO] Starting Med-Gemma inference script")


    # Counters for debugging

    total_entries = 0
    total_images_processed = 0
    total_questions = 0
    total_forward_passes = 0
    missing_images = 0

    # 1. Load JSON file
    json_path = "/path/to/json/file/chexpert_qa_long.json"
    print(f"[INFO] Loading JSON from: {json_path}")

    with open(json_path, "r") as f:
        entries = json.load(f)

    print(f"[INFO] Loaded {len(entries)} entries from JSON")

    # 2. Load Med-Gemma from local path
    model_dir = "/load/model/from/local/path"
    print(f"[INFO] Loading model from: {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    model_device = next(model.parameters()).device
    print(f"[INFO] Model loaded on device: {model_device}")

    # debug checks

   
    print("[DEBUG] torch.cuda.is_available():", torch.cuda.is_available())
    print("[DEBUG] torch.cuda.device_count():", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("[DEBUG] torch.cuda.current_device():", torch.cuda.current_device())
        print("[DEBUG] GPU name:", torch.cuda.get_device_name(0))
    else:
        print("[DEBUG] No CUDA device available — running on CPU")


    all_results = {}

    # 3. Loop over ALL entries
    for idx, entry in enumerate(entries):

        total_entries += 1

        image_path = entry["image_path"]

        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            missing_images += 1
            continue

        patient_id = extract_patient_id(image_path)
        image_id = extract_image_id(image_path)

        print(f"\n[DEBUG] Processing entry {idx + 1}/{len(entries)}")
        print(f"[DEBUG] Patient: {patient_id} | Image: {image_id}")

        image = Image.open(image_path).convert("RGB")
        total_images_processed += 1

        questions = flatten_questions(entry)
        num_questions = len(questions)
        total_questions += num_questions

        print(f"[DEBUG] Found {num_questions} questions for this image")

        if num_questions == 0:
            print("[WARN] No questions found for this entry")

        if patient_id not in all_results:
            all_results[patient_id] = {
                "sex": entry.get("sex", "UNKNOWN"),
                "No Finding": entry.get("No Finding", None),
                "images": {}
            }

        if image_id not in all_results[patient_id]["images"]:
            all_results[patient_id]["images"][image_id] = {
                "predictions": []
            }

        # loop over questions 

        for q_idx, q in enumerate(questions):

            print(f"[DEBUG]   Q {q_idx + 1}/{num_questions}: {q['finding']}")

            #change this prompt as need for the different options 
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": (
                                "Image type: Chest X-ray\n"
                                    "Decision rule:\n"
                                    "Image type: Chest X-ray\n"
                                        "Decision rule:\n"
                                        "- Assign 1 if the finding is clearly present\n"
                                        "- Assign 0 if absent \n"
                                        "- Assign 2 if you do not have the information needed to answer this question\n\n"
                                        "Output format (mandatory):\n"
                                        "Final Answer: X\n\n"
                                        f"Finding to evaluate: {q['question']}"          
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
            ).to(model_device)

            # ---- MODEL FORWARD PASS ----
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )
            total_forward_passes += 1

            generated_ids = output[0][inputs["input_ids"].shape[-1]:]
            answer_text = processor.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()

            answer = extract_answer(answer_text)

            print(f"[DEBUG]     Model raw output: '{answer_text}' → Parsed: {answer}")

            all_results[patient_id]["images"][image_id]["predictions"].append({
                "category": q["finding"],
                "question": q["question"],
                "model_answer": answer,
                "expected_answer": q["answer"]
            })

    
    # summary of debug logs 
    print("\n[INFO] ===== RUN SUMMARY =====")
    print(f"[INFO] Total JSON entries:        {total_entries}")
    print(f"[INFO] Images processed:          {total_images_processed}")
    print(f"[INFO] Missing images:            {missing_images}")
    print(f"[INFO] Total questions:           {total_questions}")
    print(f"[INFO] Total model forward calls: {total_forward_passes}")
    if total_images_processed > 0:
        print(f"[INFO] Avg questions / image:     {total_questions / total_images_processed:.2f}")

    # save results 
    output_dir = "/pathway/to/save/output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"MG_normal_3{PROMPT_NAME}.json")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[INFO] Results saved to: {output_path}")
    print("[INFO] Script finished successfully")


if __name__ == "__main__":
    main()




