import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os

MODEL_PATH = "model/composition_model_binary.pkl"
SCALER_PATH = "model/scaler_binary.pkl"
ENCODER_PATH = "model/label_encoder_binary.pkl"

FEATURE_COLUMNS = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "colorfulness",
    "rule_of_thirds_score",
    "symmetry_score",
    "aspect_ratio"
]


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


def extract_features_from_image(pil_image):
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    original_h, original_w = image_bgr.shape[:2]
    aspect_ratio = original_w / original_h

    image_resized = cv2.resize(image_bgr, (224, 224))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    features = {
        "brightness": calculate_brightness(gray),
        "contrast": calculate_contrast(gray),
        "sharpness": calculate_sharpness(gray),
        "edge_density": calculate_edge_density(gray),
        "colorfulness": calculate_colorfulness(image_resized),
        "rule_of_thirds_score": calculate_rule_of_thirds_score(gray),
        "symmetry_score": calculate_symmetry_score(gray),
        "aspect_ratio": aspect_ratio
    }

    return features


def generate_recommendations(features, prediction_label):
    recommendations = []

    if prediction_label == "baik":
        recommendations.append("Komposisi foto sudah tergolong baik berdasarkan fitur visual yang dianalisis.")
        recommendations.append("Pertahankan penempatan subjek, keseimbangan visual, dan ketajaman gambar.")
        return recommendations

    if features["rule_of_thirds_score"] < 0.25:
        recommendations.append(
            "Posisi elemen visual utama kurang dekat dengan area rule of thirds. "
            "Coba tempatkan subjek mendekati garis sepertiga gambar."
        )

    if features["symmetry_score"] < 0.02:
        recommendations.append(
            "Keseimbangan visual kiri dan kanan masih rendah. "
            "Pertimbangkan mengatur ulang posisi subjek atau melakukan cropping."
        )

    if features["sharpness"] < 500:
        recommendations.append(
            "Ketajaman gambar relatif rendah. "
            "Pastikan fokus berada pada subjek utama saat mengambil foto."
        )

    if features["brightness"] < 70:
        recommendations.append(
            "Foto terlihat cukup gelap. "
            "Pertimbangkan pencahayaan tambahan atau pengaturan exposure yang lebih baik."
        )

    if features["brightness"] > 190:
        recommendations.append(
            "Foto terlihat terlalu terang. "
            "Pertimbangkan mengurangi exposure agar detail visual tidak hilang."
        )

    if features["contrast"] < 35:
        recommendations.append(
            "Kontras gambar relatif rendah. "
            "Coba tingkatkan perbedaan terang-gelap agar subjek lebih menonjol."
        )

    if features["edge_density"] < 0.03:
        recommendations.append(
            "Detail visual pada gambar relatif sedikit. "
            "Coba dekatkan kamera ke subjek atau pilih sudut yang menampilkan elemen utama lebih jelas."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Komposisi terdeteksi perlu perbaikan, tetapi fitur dasar tidak menunjukkan masalah dominan. "
            "Coba perhatikan kembali posisi subjek, latar belakang, dan ruang kosong."
        )

    return recommendations


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder


st.set_page_config(
    page_title="Penilaian Komposisi Fotografi",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Sistem Penilaian Komposisi Fotografi Digital")
st.write(
    "Aplikasi ini menilai kualitas komposisi foto menggunakan model machine learning "
    "dan memberikan rekomendasi perbaikan berbasis aturan visual."
)

if not os.path.exists(MODEL_PATH):
    st.error("File model belum ditemukan. Pastikan model sudah dilatih dan tersimpan di folder model.")
    st.stop()

model, scaler, label_encoder = load_model()

uploaded_file = st.file_uploader(
    "Upload foto",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Foto yang Diupload")
    st.image(image, use_container_width=True)

    features = extract_features_from_image(image)

    feature_array = np.array([[features[col] for col in FEATURE_COLUMNS]])
    feature_scaled = scaler.transform(feature_array)

    prediction = model.predict(feature_scaled)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(feature_scaled)[0]
    confidence = np.max(probabilities) * 100

    st.subheader("Hasil Penilaian")

    if prediction_label == "baik":
        st.success(f"Komposisi foto diprediksi: BAIK")
    else:
        st.warning(f"Komposisi foto diprediksi: PERLU PERBAIKAN")

    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("Fitur Visual yang Diekstraksi")

    st.write({
        "Brightness": round(features["brightness"], 2),
        "Contrast": round(features["contrast"], 2),
        "Sharpness": round(features["sharpness"], 2),
        "Edge Density": round(features["edge_density"], 4),
        "Colorfulness": round(features["colorfulness"], 2),
        "Rule of Thirds Score": round(features["rule_of_thirds_score"], 4),
        "Symmetry Score": round(features["symmetry_score"], 4),
        "Aspect Ratio": round(features["aspect_ratio"], 2)
    })

    st.subheader("Rekomendasi Perbaikan Visual")

    recommendations = generate_recommendations(features, prediction_label)

    for rec in recommendations:
        st.write(f"- {rec}")

    st.caption(
        "Catatan: Hasil prediksi bersifat bantuan awal. Penilaian komposisi fotografi tetap memiliki unsur subjektif."
    )