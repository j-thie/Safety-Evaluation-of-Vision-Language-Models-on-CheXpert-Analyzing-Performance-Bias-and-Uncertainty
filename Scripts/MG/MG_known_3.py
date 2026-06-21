import json
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import re

# debug helper 
DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


# data extraction 

def flatten_questions(entry):
    question_list = []
    ignore_keys = ["image_path", "sex", "No Finding", "patient_id", "study_id", "view"]

    log(f"Flattening questions for entry with image: {entry.get('image_path')}")

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

    log(f"Extracted {len(question_list)} questions")
    return question_list


def extract_answer(text):
    log(f"Raw model output: '{text}'")

    text = text.strip()

    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1].strip()

    # Find first standalone 0/1/2 anywhere in remaining text
    match = re.search(r'\b([012])\b', text)

    if match:
        return match.group(1)

    log("Could not extract valid answer")
    return "INVALID"


def extract_patient_id(image_path):
    for part in image_path.split("/"):
        if part.startswith("patient"):
            return part
    return "UNKNOWN_PATIENT"


def extract_image_id(image_path):
    return os.path.basename(image_path)


# ---------------------------------------------------------

PROMPT_NAME = "prompt_f" # change this as necessary for the appropriate prompt option 

def main():

    log("script start")

    # Load JSON
    json_path = "/path/to/json/file/chexpert_qa_long.json"
    log(f"Loading JSON from {json_path}")

    with open(json_path, "r") as f:
        entries = json.load(f)

    log(f"Loaded {len(entries)} entries")

    # Load model
    model_dir = "/load/model/from/local/path"
    log(f"Loading model from {model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    log(f"Model loaded on device: {next(model.parameters()).device}")

    all_results = {}

    #  main loop 
    for idx, entry in enumerate(entries):
        log(f"\nProcessing entry {idx+1}/{len(entries)}")

        image_path = entry["image_path"]
        log(f"Image path: {image_path}")

        if not os.path.exists(image_path):
            log(f" Missing image: {image_path}")
            continue

        patient_id = extract_patient_id(image_path)
        image_id = extract_image_id(image_path)

        log(f"Patient ID: {patient_id}, Image ID: {image_id}")

        questions = flatten_questions(entry)

        if patient_id not in all_results:
            log(f"Creating new patient entry: {patient_id}")
            all_results[patient_id] = {
                "sex": entry.get("sex", "UNKNOWN"),
                "No Finding": entry.get("No Finding", None),
                "images": {}
            }

        if image_id not in all_results[patient_id]["images"]:
            log(f"Creating new image entry: {image_id}")
            all_results[patient_id]["images"][image_id] = {
                "predictions": []
            }

        # question loop 
        for q_idx, q in enumerate(questions):
            log(f"\nQuestion {q_idx+1}/{len(questions)}")
            log(f"Category: {q['finding']}")
            log(f"Question: {q['question']}")

            #change this prompt as need for the different options 
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a medical image classification task.\\n"
                                "Evaluate whether the listed finding is visible in the image.\\n\\n"
                                "Return ONLY the following format:\\n"
                                "Final Answer: X\\n\\n"
                                "Where:\\n"
                                "1 = Yes (finding present)\\n"
                                "0 = No  (finding absent)\\n"
                                "2 = I do not have the information needed to answer this question\\n\\n"
                                "The should be an  image is a chest X-ray, but the image is not given.\\n"
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

            log("Tokenizing input...")

            inputs = processor(
                text=text,
                return_tensors="pt"
            ).to(model.device)

            log(f"Input shape: {inputs['input_ids'].shape}")

            output = model.generate(
                **inputs,
		do_sample=False
            )

            log("Model generation complete")

            generated_ids = output[0][inputs["input_ids"].shape[-1]:]

            answer_text = processor.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()

            answer = extract_answer(answer_text)

            log(f"Final parsed answer: {answer}")
            log(f"Expected answer: {q['answer']}")

            all_results[patient_id]["images"][image_id]["predictions"].append({
                "category": q["finding"],
                "question": q["question"],
                "model_answer": answer,
                "expected_answer": q["answer"]
            })

    #  print results 
    log("\nPrinting Results")

    for patient_id, pdata in all_results.items():
        print(f"PATIENT: {patient_id}")
        print(f"Sex: {pdata['sex']}")
        print(f"No Finding: {pdata['No Finding']}")

        for image_id, results in pdata["images"].items():
            print(f"\n--- IMAGE: {image_id} ---\n")

            for r in results["predictions"]:
                print(f"[{r['category']}]")
                print(f"Q: {r['question']}")
                print(f"Model answer:    {r['model_answer']}")
                print(f"Expected answer: {r['expected_answer']}")
                print()

        print("\n")

    # save 
    output_dir = "/pathway/to/save/output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"MG_known_3{PROMPT_NAME}.json")

    log(f"Saving results to {output_path}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log("script done")


if __name__ == "__main__":
    main()

