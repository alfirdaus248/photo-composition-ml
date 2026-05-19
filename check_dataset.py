import os
import json

DATASET_DIR = "dataset/CADB_Dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
SCORE_FILE = os.path.join(DATASET_DIR, "composition_scores.json")

print("Cek folder dataset...")

print("Dataset dir ada:", os.path.exists(DATASET_DIR))
print("Folder images ada:", os.path.exists(IMAGE_DIR))
print("File score ada:", os.path.exists(SCORE_FILE))

if os.path.exists(IMAGE_DIR):
    images = os.listdir(IMAGE_DIR)
    print("Jumlah file gambar:", len(images))
    print("Contoh nama gambar:", images[:5])

if os.path.exists(SCORE_FILE):
    with open(SCORE_FILE, "r") as f:
        scores = json.load(f)

    print("Jumlah data score:", len(scores))

    first_key = list(scores.keys())[0]
    print("Contoh ID gambar:", first_key)
    print("Contoh score:", scores[first_key])