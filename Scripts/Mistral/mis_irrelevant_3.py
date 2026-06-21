import json
import os
import base64
from io import BytesIO
from PIL import Image
from vllm import LLM, SamplingParams
import re
import random
from pathlib import Path

# image handling 

def encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# data extraction 

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
                    "answer": q["answer"],
                })
    return question_list


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

def extract_patient_id(image_path):
    for part in image_path.split("/"):
        if part.startswith("patient"):
            return part
    return "UNKNOWN_PATIENT"

def extract_image_id(image_path):
    return os.path.basename(image_path)



PROMPT_NAME = "prompt_f" # change this based on which prompt is being used 

def main():

    print("[INFO] Starting inference script")

    total_entries = 0
    total_images_processed = 0
    total_questions = 0
    total_forward_passes = 0
    missing_images = 0

    json_path = "/path/to/json/file/chexpert_qa_long.json"
    print(f"[INFO] Loading JSON: {json_path}")

    with open(json_path) as f:
        entries = json.load(f)

    imagenet_root = Path("/path/to/irrelevant/photo/dataset")
    imagenet_images = list(imagenet_root.rglob("*.JPEG"))

    print(f"[INFO] Found {len(imagenet_images)} ImageNet images")

    #load model

    llm = LLM(
        model="mistralai/Ministral-3-14B-Instruct-2512",
        tokenizer="mistralai/Ministral-3-14B-Instruct-2512",
        tokenizer_mode="mistral",
        gpu_memory_utilization=0.95,
        max_model_len=32000,
    )

    sampling_params = SamplingParams(
        max_tokens=32,
        temperature=0.0,
        top_p=1.0
    )

    all_results = {}


    # loop over entires 

    for entry in entries:
        total_entries += 1

        image_path = str(random.choice(imagenet_images))

        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            missing_images += 1
            continue

        patient_id = extract_patient_id(image_path)
        image_id = extract_image_id(image_path)

        image = load_image(image_path)
        base64_image = encode_image(image)
        total_images_processed += 1

        questions = flatten_questions(entry)
        total_questions += len(questions)

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

        # question loop 

        for q in questions:

            prompt = (
                    "This is a medical image classification task.\n"
                    "Evaluate whether the listed finding is visible in the image.\n\n"
                    "Return ONLY the following format:\n"
                    "Final Answer: X\n\n"
                    "Where:\n"
                    "1 = Yes (finding present)\n"
                    "0 = No  (finding absent)\n"
                    "2 = I do not have the information needed to answer this question\n\n"
                    "The image is a chest X-ray.\n"
                    f"Finding: {q['question']}"
                    )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }]

            outputs = llm.chat(messages, sampling_params=sampling_params)
            total_forward_passes += 1

            raw_text = outputs[0].outputs[0].text.strip()
            answer = extract_answer(raw_text)

            all_results[patient_id]["images"][image_id]["predictions"].append({
                "category": q["finding"],
                "question": q["question"],
                "model_answer": answer,
                "expected_answer": q["answer"],
            })

    # print statements 

    print("\n[INFO] run summary")
    print(f"[INFO] Total JSON entries:        {total_entries}")
    print(f"[INFO] Images processed:          {total_images_processed}")
    print(f"[INFO] Missing images:            {missing_images}")
    print(f"[INFO] Total questions:           {total_questions}")
    print(f"[INFO] Total model forward calls: {total_forward_passes}")
    if total_images_processed > 0:
        print(f"[INFO] Avg questions / image:     {total_questions / total_images_processed:.2f}")

    # save the results 

    output_dir = "/pathway/to/save/output"
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"mis_irrelevant_3{PROMPT_NAME}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[INFO] Results saved to: {out_path}")
    print("[INFO] Script finished successfully")


if __name__ == "__main__":
    main()
