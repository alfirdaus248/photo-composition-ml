import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

INPUT_FILE = "labels.csv"
OUTPUT_FILE = "features.csv"


def calculate_brightness(gray):
    return np.mean(gray)


def calculate_contrast(gray):
    return np.std(gray)


def calculate_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def calculate_colorfulness(image):
    # image dalam format BGR
    b, g, r = cv2.split(image.astype("float"))

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    return np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)


def calculate_rule_of_thirds_score(gray):
    h, w = gray.shape

    edges = cv2.Canny(gray, 100, 200)

    x_lines = [w // 3, 2 * w // 3]
    y_lines = [h // 3, 2 * h // 3]

    tolerance_x = max(1, w // 20)
    tolerance_y = max(1, h // 20)

    mask = np.zeros_like(edges)

    for x in x_lines:
        mask[:, max(0, x - tolerance_x):min(w, x + tolerance_x)] = 255

    for y in y_lines:
        mask[max(0, y - tolerance_y):min(h, y + tolerance_y), :] = 255

    thirds_edges = np.sum((edges > 0) & (mask > 0))
    total_edges = np.sum(edges > 0)

    if total_edges == 0:
        return 0

    return thirds_edges / total_edges


def calculate_symmetry_score(gray):
    h, w = gray.shape

    left = gray[:, :w // 2]
    right = gray[:, w - w // 2:]
    right_flipped = cv2.flip(right, 1)

    min_width = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_width]
    right_flipped = right_flipped[:, :min_width]

    difference = np.mean(np.abs(left.astype("float") - right_flipped.astype("float")))

    return 1 / (1 + difference)


def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = {
        "brightness": calculate_brightness(gray),
        "contrast": calculate_contrast(gray),
        "sharpness": calculate_sharpness(gray),
        "edge_density": calculate_edge_density(gray),
        "colorfulness": calculate_colorfulness(image),
        "rule_of_thirds_score": calculate_rule_of_thirds_score(gray),
        "symmetry_score": calculate_symmetry_score(gray),
        "aspect_ratio": image.shape[1] / image.shape[0]
    }

    return features


df = pd.read_csv(INPUT_FILE)

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row["image_path"]

    features = extract_features(image_path)

    if features is not None:
        features["image_name"] = row["image_name"]
        features["mean_score"] = row["mean_score"]
        features["label"] = row["label"]

        rows.append(features)

features_df = pd.DataFrame(rows)

features_df.to_csv(OUTPUT_FILE, index=False)

print("Ekstraksi fitur selesai.")
print("Jumlah data fitur:", len(features_df))
print("File berhasil dibuat:", OUTPUT_FILE)
print(features_df.head())