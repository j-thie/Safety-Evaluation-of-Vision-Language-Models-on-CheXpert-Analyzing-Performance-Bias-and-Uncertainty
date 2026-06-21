import json
import os
import base64
import logging
from io import BytesIO
from PIL import Image
from vllm import LLM, SamplingParams

# debug helper 
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

#  image helper 

def encode_image(image: Image.Image) -> str:
    logger.debug("Encoding image to base64")
    buf = BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    logger.debug(f"Encoded image size: {len(encoded)} characters")
    return encoded


def load_image(path):
    logger.debug(f"Loading image: {path}")
    img = Image.open(path)
    if img.mode != "RGB":
        logger.debug(f"Converting image to RGB from {img.mode}")
        img = img.convert("RGB")
    return img


# data extraction 

def flatten_questions(entry):
    logger.debug("Flattening questions for entry")
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

    logger.debug(f"Extracted {len(question_list)} questions")
    return question_list


def extract_binary_answer(text):
    logger.debug(f"Raw model output: {text}")
    text = text.strip()

    if "Final Answer:" in text:
        text = text.split("Final Answer:")[-1]

    if "1" in text:
        return "1"
    if "0" in text:
        return "0"

    logger.warning(f"Invalid answer format: {text}")
    return "INVALID"


def extract_patient_id(image_path):
    for part in image_path.split("/"):
        if part.startswith("patient"):
            return part
    logger.warning(f"Could not extract patient_id from: {image_path}")
    return "UNKNOWN_PATIENT"


def extract_image_id(image_path):
    return os.path.basename(image_path)


PROMPT_NAME = "prompt_f" # change this as necessary for the appropriate prompt option 


def main():
    logger.info("Starting inference script")

    total_entries = 0
    total_images_processed = 0
    total_questions = 0
    total_forward_passes = 0
    missing_images = 0

    json_path = "/path/to/json/file/chexpert_qa_long.json"
    logger.info(f"Loading JSON: {json_path}")

    with open(json_path) as f:
        entries = json.load(f)

    logger.info(f"Loaded {len(entries)} entries")

    # model 
    logger.info("Loading model...")
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

    #  main loop 
    for entry_idx, entry in enumerate(entries):
        total_entries += 1
        logger.debug(f"Processing entry {entry_idx}")

        image_path = entry["image_path"]

        if not os.path.exists(image_path):
            logger.warning(f"Missing image: {image_path}")
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
            logger.debug(f"Creating new patient entry: {patient_id}")
            all_results[patient_id] = {
                "sex": entry.get("sex", "UNKNOWN"),
                "No Finding": entry.get("No Finding", None),
                "images": {}
            }

        if image_id not in all_results[patient_id]["images"]:
            logger.debug(f"Creating new image entry: {image_id}")
            all_results[patient_id]["images"][image_id] = {
                "predictions": []
            }

        # question loop 
        for q_idx, q in enumerate(questions):
            logger.debug(f"Processing question {q_idx}: {q['question']}")

            # change prompt here as needed 
            prompt = (
                        "This is a medical image classification task.\\n"
                        "Evaluate whether the listed finding is visible in the image.\\n\\n"
                        "Return ONLY the following format:\\n"
                        "Final Answer: X\\n\\n"
                        "Where:\\n"
                        "1 = Yes (finding present)\\n"
                        "0 = No  (finding absent)\\n"
                        "The should be an  image is a chest X-ray, but the image is not given.\\n"
                        f"Finding: {q['question']}"
                    	    )

            logger.debug(f"Prompt: {prompt}")

            messages = [{
                "role": "user",
                "content": prompt
            }]

            outputs = llm.chat(messages, sampling_params=sampling_params)
            total_forward_passes += 1

            raw_text = outputs[0].outputs[0].text.strip()
            answer = extract_binary_answer(raw_text)

            logger.debug(f"Model answer: {answer}")

            all_results[patient_id]["images"][image_id]["predictions"].append({
                "category": q["finding"],
                "question": q["question"],
                "model_answer": answer,
                "expected_answer": q["answer"],
            })

    # summary 
    logger.info("run summary")
    logger.info(f"Total JSON entries:        {total_entries}")
    logger.info(f"Images processed:          {total_images_processed}")
    logger.info(f"Missing images:            {missing_images}")
    logger.info(f"Total questions:           {total_questions}")
    logger.info(f"Total model forward calls: {total_forward_passes}")

    if total_images_processed > 0:
        logger.info(f"Avg questions / image: {total_questions / total_images_processed:.2f}")

    # save 
    output_dir = "/pathway/to/save/output"
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"mis_known_2_{PROMPT_NAME}.json")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to: {out_path}")
    logger.info("Script finished successfully")


if __name__ == "__main__":
    main()
