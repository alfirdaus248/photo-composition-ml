import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import textwrap

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "model/mobilenet_composition_model.h5"
LABEL_PATH = "model/mobilenet_label_classes.npy"

IMAGE_SIZE = 224


# =========================
# FEATURE EXTRACTION
# =========================

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


def extract_visual_features(pil_image):
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    original_h, original_w = image_bgr.shape[:2]
    aspect_ratio = original_w / original_h

    image_resized = cv2.resize(image_bgr, (224, 224))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    return {
        "brightness": calculate_brightness(gray),
        "contrast": calculate_contrast(gray),
        "sharpness": calculate_sharpness(gray),
        "edge_density": calculate_edge_density(gray),
        "colorfulness": calculate_colorfulness(image_resized),
        "rule_of_thirds_score": calculate_rule_of_thirds_score(gray),
        "symmetry_score": calculate_symmetry_score(gray),
        "aspect_ratio": aspect_ratio
    }


def prepare_image_for_model(pil_image):
    image = pil_image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    return image_array


# =========================
# RECOMMENDATION SYSTEM
# =========================

def generate_recommendations(features, prediction_label):
    good_points = []
    improvements = []

    # Ambil fitur agar kode lebih mudah dibaca
    brightness = features["brightness"]
    contrast = features["contrast"]
    sharpness = features["sharpness"]
    edge_density = features["edge_density"]
    colorfulness = features["colorfulness"]
    thirds = features["rule_of_thirds_score"]
    symmetry = features["symmetry_score"]
    aspect_ratio = features["aspect_ratio"]

    # =========================
    # ASPEK YANG SUDAH BAIK
    # =========================

    if sharpness >= 3000:
        good_points.append(
            "Ketajaman gambar sangat baik. Detail visual terlihat jelas sehingga subjek atau elemen utama lebih mudah dikenali."
        )
    elif sharpness >= 1000:
        good_points.append(
            "Ketajaman gambar cukup baik. Detail visual masih terlihat jelas dan mendukung keterbacaan foto."
        )

    if 80 <= brightness <= 170:
        good_points.append(
            "Pencahayaan berada pada rentang yang cukup seimbang, tidak terlalu gelap dan tidak terlalu terang."
        )

    if contrast >= 60:
        good_points.append(
            "Kontras gambar cukup kuat sehingga pemisahan antara objek, latar, dan elemen visual terlihat lebih jelas."
        )
    elif contrast >= 45:
        good_points.append(
            "Kontras gambar cukup baik dan membantu objek utama terlihat lebih menonjol dari latar."
        )

    is_minimal_architectural_style = (
        colorfulness < 15
        and contrast > 60
        and sharpness > 3000
        and thirds > 0.35
    )

    if is_minimal_architectural_style:
        good_points.append(
            "Foto memiliki karakter visual minimalis dengan warna rendah, kontras kuat, dan garis arsitektural yang tegas. "
            "Gaya ini dapat mendukung kesan dramatis pada foto arsitektur."
        )

    if colorfulness >= 55:
        good_points.append(
            "Warna pada foto cukup hidup dan memberi daya tarik visual yang kuat."
        )
    elif colorfulness >= 35:
        good_points.append(
            "Warna pada foto cukup bervariasi dan mendukung suasana visual gambar."
        )

    if thirds >= 0.40:
        good_points.append(
            "Elemen visual cukup kuat berada di area rule of thirds, sehingga komposisi memiliki potensi terlihat dinamis."
        )
    elif thirds >= 0.30:
        good_points.append(
            "Sebagian elemen visual sudah cukup dekat dengan area rule of thirds."
        )

    if 0.020 <= symmetry <= 0.045:
        good_points.append(
            "Distribusi visual kiri dan kanan masih cukup seimbang tanpa terlihat terlalu kaku atau terlalu simetris."
        )

    if 0.05 <= edge_density <= 0.14:
        good_points.append(
            "Kepadatan detail visual berada pada tingkat yang cukup nyaman, sehingga foto tidak terlihat terlalu kosong atau terlalu ramai."
        )

    # =========================
    # REKOMENDASI TEKNIS DASAR
    # =========================

    if sharpness < 300:
        improvements.append(
            "Ketajaman gambar sangat rendah. Foto terlihat blur, sehingga subjek utama sulit dikenali. "
            "Gunakan fokus yang lebih tepat, stabilkan kamera, atau gunakan shutter speed lebih cepat saat mengambil gambar."
        )
    elif sharpness < 500:
        improvements.append(
            "Ketajaman gambar relatif rendah. Pastikan fokus berada pada subjek utama dan hindari gerakan kamera saat pengambilan foto."
        )
    elif sharpness < 900 and prediction_label == "perlu_perbaikan":
        improvements.append(
            "Ketajaman gambar masih dapat ditingkatkan. Detail pada subjek atau elemen penting sebaiknya dibuat lebih jelas agar foto terlihat lebih kuat."
        )

    if brightness < 50:
        improvements.append(
            "Foto terlihat sangat gelap. Detail pada subjek atau latar berpotensi hilang. "
            "Pertimbangkan pencahayaan tambahan, exposure lebih tinggi, atau pengambilan gambar pada kondisi cahaya yang lebih baik."
        )
    elif brightness < 75:
        improvements.append(
            "Foto cenderung gelap. Pertimbangkan meningkatkan exposure atau memilih arah cahaya yang lebih mendukung subjek utama."
        )
    elif brightness > 210:
        improvements.append(
            "Foto terlihat sangat terang. Beberapa detail visual berpotensi hilang karena overexposure. "
            "Pertimbangkan menurunkan exposure atau menghindari sumber cahaya yang terlalu kuat."
        )
    elif brightness > 185:
        improvements.append(
            "Foto cenderung terlalu terang. Pertimbangkan mengurangi exposure agar detail pada area terang tetap terjaga."
        )

    if contrast < 25:
        improvements.append(
            "Kontras gambar sangat rendah. Foto dapat terlihat datar dan subjek sulit dipisahkan dari latar. "
            "Coba tingkatkan perbedaan terang-gelap atau pilih pencahayaan yang lebih membentuk dimensi."
        )
    elif contrast < 38:
        improvements.append(
            "Kontras gambar relatif rendah. Tingkatkan perbedaan antara subjek dan latar agar fokus visual lebih jelas."
        )

    if colorfulness < 18 and not is_minimal_architectural_style:
        improvements.append(
            "Warna pada foto terlihat kurang kuat. Jika sesuai dengan konsep foto, pertimbangkan penyesuaian warna atau pencahayaan agar suasana visual lebih hidup."
        )
    elif colorfulness > 75 and prediction_label == "perlu_perbaikan":
        improvements.append(
            "Warna pada foto sangat mencolok. Pastikan warna yang dominan tetap mendukung subjek utama dan tidak menjadi distraksi visual."
        )

    # =========================
    # REKOMENDASI KOMPOSISI
    # =========================

    if thirds < 0.18:
        improvements.append(
            "Elemen visual utama kurang terhubung dengan area rule of thirds. "
            "Pertimbangkan menempatkan subjek sedikit ke kiri, kanan, atas, atau bawah dari pusat frame agar komposisi lebih dinamis."
        )
    elif thirds < 0.27:
        improvements.append(
            "Penempatan elemen visual masih dapat diperbaiki. Coba posisikan subjek lebih dekat ke garis atau titik sepertiga gambar."
        )

    if symmetry < 0.012:
        improvements.append(
            "Distribusi visual kiri dan kanan sangat tidak seimbang. Jika bukan bagian dari gaya artistik, lakukan cropping atau atur ulang posisi subjek agar komposisi lebih stabil."
        )
    elif symmetry < 0.020:
        improvements.append(
            "Keseimbangan visual kiri dan kanan masih rendah. Pertimbangkan cropping atau pengaturan ulang posisi subjek agar frame terasa lebih seimbang."
        )

    if edge_density < 0.025:
        improvements.append(
            "Foto memiliki detail visual yang sangat sedikit. Jika foto terasa kosong, coba dekatkan kamera ke subjek atau tambahkan elemen pendukung yang relevan."
        )
    elif edge_density < 0.040 and prediction_label == "perlu_perbaikan":
        improvements.append(
            "Detail visual pada gambar relatif sedikit. Coba pilih sudut pengambilan yang memperjelas bentuk, garis, atau tekstur dari subjek utama."
        )

    if edge_density > 0.22:
        improvements.append(
            "Foto memiliki kepadatan detail visual yang sangat tinggi. Background atau elemen sekitar berpotensi terlalu ramai dan mengalihkan perhatian dari subjek utama. "
            "Pertimbangkan cropping, sudut pengambilan yang lebih sederhana, atau depth of field yang lebih dangkal."
        )
    elif edge_density > 0.14 and prediction_label == "perlu_perbaikan":
        improvements.append(
            "Foto memiliki cukup banyak detail visual. Pastikan background, foreground, atau elemen sekitar tidak bersaing terlalu kuat dengan subjek utama."
        )

    # =========================
    # REKOMENDASI KONTEKSTUAL BERDASARKAN KOMBINASI FITUR
    # =========================

    if prediction_label == "perlu_perbaikan" and sharpness >= 1000 and contrast >= 45 and thirds >= 0.30:
        improvements.append(
            "Secara teknis foto memiliki beberapa aspek yang baik, seperti ketajaman, kontras, atau penempatan elemen. "
            "Namun model tetap menilai foto perlu perbaikan, sehingga masalah kemungkinan berasal dari faktor komposisi yang lebih kompleks seperti fokus visual yang kurang jelas, distraksi background, atau subjek yang kurang dominan."
        )

    if prediction_label == "perlu_perbaikan" and thirds >= 0.35:
        improvements.append(
            "Walaupun beberapa elemen sudah berada dekat area rule of thirds, fokus utama foto masih dapat dibuat lebih jelas. "
            "Kurangi elemen foreground atau background yang tidak mendukung cerita visual."
        )

    if prediction_label == "perlu_perbaikan" and colorfulness > 40:
        improvements.append(
            "Warna pada foto cukup kuat, tetapi pastikan warna yang paling mencolok tidak menarik perhatian menjauh dari subjek utama."
        )

    if prediction_label == "perlu_perbaikan" and edge_density > 0.10 and thirds >= 0.30:
        improvements.append(
            "Foto memiliki struktur komposisi yang cukup aktif, tetapi juga mengandung banyak elemen visual. "
            "Coba tentukan satu pusat perhatian utama agar mata penonton tidak berpindah ke terlalu banyak objek."
        )

    if prediction_label == "perlu_perbaikan" and aspect_ratio > 1.65:
        improvements.append(
            "Foto memiliki rasio yang cukup lebar. Pastikan ruang horizontal benar-benar mendukung cerita visual. "
            "Jika ada area kosong atau distraksi di sisi kiri/kanan, pertimbangkan cropping agar komposisi lebih fokus."
        )

    if prediction_label == "perlu_perbaikan" and aspect_ratio < 0.85:
        improvements.append(
            "Foto memiliki orientasi vertikal yang cukup kuat. Pastikan subjek utama memiliki ruang yang cukup dan tidak terlalu tertekan oleh bagian atas atau bawah frame."
        )

    if prediction_label == "baik" and edge_density > 0.20:
        improvements.append(
            "Komposisi dinilai baik, tetapi gambar memiliki detail yang cukup padat. Tetap perhatikan agar background tidak terlalu mendominasi subjek utama."
        )

    if prediction_label == "baik" and sharpness < 800:
        improvements.append(
            "Komposisi dinilai baik, tetapi ketajaman masih dapat ditingkatkan agar hasil foto terlihat lebih bersih dan profesional."
        )

    if prediction_label == "baik" and symmetry < 0.015:
        improvements.append(
            "Komposisi dinilai baik, tetapi keseimbangan visual kiri dan kanan masih dapat disempurnakan melalui cropping ringan."
        )

    if prediction_label == "baik" and edge_density > 0.12:
        improvements.append(
            "Komposisi foto sudah cukup baik, tetapi terdapat beberapa elemen detail di sekitar frame. "
            "Jika ingin hasil yang lebih fokus, pertimbangkan cropping ringan pada area yang tidak mendukung subjek utama."
        )

    if prediction_label == "baik" and symmetry < 0.018:
        improvements.append(
            "Komposisi terlihat asimetris, tetapi masih dapat diterima secara visual. "
            "Untuk hasil yang lebih rapi, pertimbangkan cropping ringan agar distribusi elemen kiri dan kanan terasa lebih seimbang."
        )

    # =========================
    # BATASI JUMLAH REKOMENDASI AGAR TIDAK TERLALU PANJANG
    # =========================
    # Prioritaskan rekomendasi yang paling penting berdasarkan urutan kemunculan.
    if len(improvements) > 6:
        improvements = improvements[:6]

    if len(good_points) > 5:
        good_points = good_points[:5]

    # =========================
    # FALLBACK
    # =========================

    if prediction_label == "baik" and len(improvements) == 0:
        improvements.append(
            "Tidak ada perbaikan dominan yang terdeteksi. Komposisi foto sudah cukup baik berdasarkan model dan fitur visual pendukung."
        )

    if prediction_label == "perlu_perbaikan" and len(improvements) == 0:
        improvements.append(
            "Model menilai foto perlu perbaikan, tetapi fitur visual dasar tidak menunjukkan masalah dominan. "
            "Hal ini dapat terjadi karena aspek komposisi yang lebih kompleks, seperti cerita visual yang kurang jelas, subjek kurang dominan, atau elemen visual yang saling bersaing."
        )
        improvements.append(
            "Coba evaluasi kembali framing, posisi subjek, ruang kosong, dan hubungan antara foreground-background agar perhatian penonton lebih terarah."
        )

    if len(good_points) == 0:
        good_points.append(
            "Belum ada aspek visual dominan yang terdeteksi sebagai keunggulan utama berdasarkan fitur yang dianalisis."
        )

    return good_points, improvements


def get_confidence_level(confidence):
    if confidence >= 85:
        return "Tinggi"
    elif confidence >= 70:
        return "Sedang"
    else:
        return "Rendah"


def get_prediction_summary(prediction_label):
    if prediction_label == "baik":
        return (
            "Secara umum, foto ini memiliki komposisi yang cukup baik. "
            "Beberapa fitur visual seperti ketajaman, kontras, warna, atau penempatan elemen visual "
            "mendukung hasil penilaian model."
        )

    return (
        "Foto ini masih memiliki potensi perbaikan dari sisi komposisi atau kualitas visual. "
        "Beberapa aspek seperti fokus, kepadatan background, pencahayaan, atau keseimbangan visual "
        "dapat diperhatikan kembali."
    )


# =========================
# MODEL LOADING
# =========================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABEL_PATH, allow_pickle=True)
    return model, labels


# =========================
# UI CONFIG
# =========================

st.set_page_config(
    page_title="Penilaian Komposisi Fotografi",
    page_icon="📷",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: white;
        padding: 2rem;
        border-radius: 22px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.16);
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #d1d5db;
        line-height: 1.6;
        max-width: 900px;
    }

    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .good-result {
        border-left: 8px solid #16a34a;
    }

    .warning-result {
        border-left: 8px solid #f59e0b;
    }

    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: #111827;
    }

    .result-desc {
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    .confidence-number {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }

    .metric-label {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 750;
    }

    .footer-note {
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================

st.markdown("""
<div class="hero-card">
    <div class="hero-title">📷 Sistem Penilaian Komposisi Fotografi Digital</div>
    <div class="hero-subtitle">
        Aplikasi ini menggunakan model MobileNetV2 berbasis transfer learning untuk menilai kualitas komposisi foto,
        menampilkan fitur visual pendukung, dan memberikan rekomendasi perbaikan berbasis aturan visual.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================
# CHECK FILES
# =========================

if not os.path.exists(MODEL_PATH):
    st.error("Model MobileNetV2 belum ditemukan. Pastikan file model tersedia di folder model.")
    st.stop()

if not os.path.exists(LABEL_PATH):
    st.error("File label MobileNetV2 belum ditemukan.")
    st.stop()


model, labels = load_model()


# =========================
# UPLOAD AREA
# =========================

with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Upload foto untuk dianalisis",
        type=None,
        help="Gunakan foto JPG, JPEG, atau PNG."
    )


# =========================
# MAIN APP
# =========================

if uploaded_file is None:
    st.info("Silakan upload foto terlebih dahulu untuk melihat hasil penilaian komposisi.")
    st.stop()

allowed_extensions = ["jpg", "jpeg", "png"]

file_extension = uploaded_file.name.split(".")[-1].lower()

if file_extension not in allowed_extensions:
    st.error("Format file tidak didukung. Silakan upload file dengan format JPG, JPEG, atau PNG.")

    if st.button("Upload ulang"):
        st.rerun()

    st.stop()


image = Image.open(uploaded_file)

model_input = prepare_image_for_model(image)
features = extract_visual_features(image)

pred_prob = model.predict(model_input, verbose=0)[0][0]

if pred_prob >= 0.5:
    pred_index = 1
    confidence = pred_prob * 100
else:
    pred_index = 0
    confidence = (1 - pred_prob) * 100

prediction_label = str(labels[pred_index])
confidence_level = get_confidence_level(confidence)

override_reason = None

if features["sharpness"] < 500:
    prediction_label = "perlu_perbaikan"
    override_reason = (
        "Foto memiliki ketajaman yang sangat rendah sehingga hasil akhir dikoreksi menjadi perlu perbaikan."
    )

# Rule correction untuk foto yang secara teknis sangat kuat
if (
    prediction_label == "perlu_perbaikan"
    and confidence < 85
    and features["sharpness"] > 3000
    and 80 <= features["brightness"] <= 170
    and features["contrast"] > 55
    and features["rule_of_thirds_score"] > 0.35
):
    prediction_label = "baik"
    override_reason = (
        "Foto memiliki kualitas teknis dan komposisi dasar yang kuat, sehingga hasil akhir dikoreksi menjadi baik. "
        "Beberapa rekomendasi tetap diberikan sebagai perbaikan ringan."
    )

elif (
    prediction_label == "perlu_perbaikan"
    and confidence < 90
    and features["sharpness"] > 3000
    and features["contrast"] > 65
    and features["rule_of_thirds_score"] > 0.40
    and features["edge_density"] < 0.15
):
    prediction_label = "baik"
    override_reason = (
        "Foto memiliki karakter arsitektural yang kuat dengan ketajaman tinggi, kontras kuat, dan komposisi garis yang jelas. "
        "Hasil akhir dikoreksi menjadi baik, dengan catatan perbaikan ringan pada exposure atau keseimbangan visual."
    )

good_points, improvements = generate_recommendations(features, prediction_label)
summary_text = get_prediction_summary(prediction_label)


# =========================
# TOP RESULT LAYOUT
# =========================

left_col, right_col = st.columns([1.1, 1])

with left_col:
    with st.container(border=True):
        st.subheader("Foto yang Diupload")
        st.image(image, use_container_width=True)

with right_col:
    st.subheader("Hasil Penilaian")

    if prediction_label == "baik":
        st.markdown(f"""
        <div class="result-card good-result">
            <div class="result-title">✅ BAIK</div>
            <div class="result-desc">Komposisi foto diprediksi memiliki kualitas yang cukup baik.</div>
            <div class="confidence-number">{confidence:.2f}%</div>
            <div class="small-note">Tingkat keyakinan model: <b>{confidence_level}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card warning-result">
            <div class="result-title">⚠️ PERLU PERBAIKAN</div>
            <div class="result-desc">Foto masih memiliki potensi perbaikan dari sisi komposisi atau kualitas visual.</div>
            <div class="confidence-number">{confidence:.2f}%</div>
            <div class="small-note">Tingkat keyakinan model: <b>{confidence_level}</b></div>
        </div>
        """, unsafe_allow_html=True)

    if override_reason is not None:
        st.info(override_reason)
        st.caption("Confidence di atas berasal dari prediksi model awal sebelum koreksi berbasis aturan visual.")

    with st.container(border=True):
        st.subheader("Ringkasan Analisis")
        st.write(summary_text)


# =========================
# FEATURE METRICS
# =========================

st.subheader("Fitur Visual Pendukung")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Brightness</div>
        <div class="metric-value">{features["brightness"]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Contrast</div>
        <div class="metric-value">{features["contrast"]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Sharpness</div>
        <div class="metric-value">{features["sharpness"]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Edge Density</div>
        <div class="metric-value">{features["edge_density"]:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

m5, m6, m7, m8 = st.columns(4)

with m5:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Colorfulness</div>
        <div class="metric-value">{features["colorfulness"]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m6:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Rule of Thirds</div>
        <div class="metric-value">{features["rule_of_thirds_score"]:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with m7:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Symmetry</div>
        <div class="metric-value">{features["symmetry_score"]:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with m8:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Aspect Ratio</div>
        <div class="metric-value">{features["aspect_ratio"]:.2f}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# RECOMMENDATION CARDS
# =========================

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    with st.container(border=True):
        st.subheader("✅ Aspek yang Sudah Baik")

        if len(good_points) == 0:
            st.write("- Belum ada aspek visual dominan yang terdeteksi sebagai keunggulan utama.")
        else:
            for item in good_points:
                st.write(f"- {item}")

with rec_col2:
    with st.container(border=True):
        st.subheader("🛠️ Rekomendasi Perbaikan")

        if len(improvements) == 0:
            st.write("- Tidak ada rekomendasi perbaikan dominan yang terdeteksi.")
        else:
            for item in improvements:
                st.write(f"- {item}")


# =========================
# DOWNLOAD REPORT
# =========================

good_points_text = "\n".join(["- " + item for item in good_points]) if good_points else "- Tidak ada aspek dominan."
improvements_text = "\n".join(["- " + item for item in improvements]) if improvements else "- Tidak ada rekomendasi dominan."

analysis_text = f"""
Hasil Penilaian: {prediction_label}
Confidence: {confidence:.2f}%
Tingkat Keyakinan Model: {confidence_level}

Ringkasan Analisis:
{summary_text}

Fitur Visual:
Brightness: {features["brightness"]:.2f}
Contrast: {features["contrast"]:.2f}
Sharpness: {features["sharpness"]:.2f}
Edge Density: {features["edge_density"]:.4f}
Colorfulness: {features["colorfulness"]:.2f}
Rule of Thirds Score: {features["rule_of_thirds_score"]:.4f}
Symmetry Score: {features["symmetry_score"]:.4f}
Aspect Ratio: {features["aspect_ratio"]:.2f}

Aspek yang Sudah Baik:
{good_points_text}

Rekomendasi:
{improvements_text}
"""

analysis_text = textwrap.dedent(analysis_text).strip()

st.download_button(
    label="📄 Download Hasil Analisis",
    data=analysis_text,
    file_name="hasil_analisis_foto.txt",
    mime="text/plain"
)

st.markdown(
    """
    <div class="footer-note">
        Catatan: Prediksi model menggunakan MobileNetV2, sedangkan rekomendasi visual dibuat berdasarkan aturan fitur visual dasar.
        Penilaian komposisi fotografi tetap memiliki unsur subjektif.
    </div>
    """,
    unsafe_allow_html=True
)