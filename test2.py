import os
import base64
import argparse
import re
import time

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
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    images_dir = os.path.join(args.root, "images", args.split)
    gt_path = os.path.join(args.root, "labels", args.gt_csv)

    df = pd.read_csv(gt_path)
    with open(args.prompt_file, "r") as f:
        prompt_text = f.read()

    # -----------------------
    # 🔍 Isolate 490th sample
    # -----------------------
    idx = 490  # 0-indexed
    row = df.iloc[idx]

    image_id = row["image_id"]
    ground_truth = row["ground_truth"]
    image_path = os.path.join(images_dir, image_id)

    print("DEBUG INDEX:", idx)
    print("Image ID:", image_id)
    print("Ground Truth:", ground_truth)
    print("Image Path:", image_path)
    print("File Size (bytes):", os.path.getsize(image_path))

    image_base64 = encode_image(image_path)

    # Dynamically determine MIME type
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".png":
        mime = "png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "jpeg"
    elif ext == ".gif":
        mime = "gif"
    elif ext == ".webp":
        mime = "webp"
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    try:
        #breakpoint()
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
                                "url": f"data:image/{mime};base64,{image_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        )

        raw_answer = response.choices[0].message.content
        prediction = parse_prediction(raw_answer)

        print("\nModel Output:")
        print(raw_answer)
        print("Parsed Prediction:", prediction)

    except Exception as e:
        print("\nAPI ERROR:")
        print(e)


if __name__ == "__main__":
    main()