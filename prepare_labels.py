import os
import json
import pandas as pd

DATASET_DIR = "dataset/CADB_Dataset"
SCORE_FILE = os.path.join(DATASET_DIR, "composition_scores.json")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")

OUTPUT_FILE = "labels.csv"


def score_to_label(mean_score):
    if mean_score < 2.5:
        return "rendah"
    elif mean_score < 3.5:
        return "sedang"
    else:
        return "tinggi"


with open(SCORE_FILE, "r") as f:
    scores = json.load(f)

rows = []

for image_name, data in scores.items():
    image_path = os.path.join(IMAGE_DIR, image_name)

    if os.path.exists(image_path):
        mean_score = data["mean"]
        label = score_to_label(mean_score)

        rows.append({
            "image_name": image_name,
            "image_path": image_path,
            "mean_score": mean_score,
            "label": label
        })

df = pd.DataFrame(rows)

print("Jumlah data:", len(df))
print("\nDistribusi label:")
print(df["label"].value_counts())

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nFile berhasil dibuat: {OUTPUT_FILE}")