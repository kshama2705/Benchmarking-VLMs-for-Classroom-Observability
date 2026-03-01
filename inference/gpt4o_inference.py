import os
import base64
import argparse
import csv
import re
import time
from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------
# Helper: encode image
# -----------------------
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -----------------------
# Helper: parse prediction
# -----------------------
def parse_prediction(text):
    """
    Extract last standalone digit 0-3.
    Defaults to 2 if no valid digit found (matching coauthor behavior).
    """
    matches = re.findall(r"\b[0-3]\b", text)
    if matches:
        return int(matches[-1])
    return 2

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--gt_csv", required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.5)

    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    images_dir = os.path.join(args.root, "images", args.split)
    gt_path = os.path.join(args.root, "labels", args.gt_csv)

    df = pd.read_csv(gt_path)
    with open(args.prompt_file, "r") as f:
        prompt_text = f.read()

    results = []

    for _, row in tqdm(df.iterrows()):
        image_id = row["image_id"]
        ground_truth = row["ground_truth"]

        image_path = os.path.join(images_dir, image_id)
        image_base64 = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=args.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )

        raw_answer = response.choices[0].message.content
        prediction = parse_prediction(raw_answer)

        results.append({
            "image_id": image_id,
            "ground_truth": ground_truth,
            "predicted": prediction,
            "raw_answer": raw_answer
        })

        time.sleep(args.sleep)

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()