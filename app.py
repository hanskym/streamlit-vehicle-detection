import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import tempfile
import os
import io

st.set_page_config(
    page_title="Vehicle Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    :root {
        --bg: #ffffff;
        --text: #111827;
        --muted: #666666;
        --card-bg: #f0f2f6;
        --accent: #FF4B4B;
        --danger: #FF4B4B;
        --danger-variant: #FF6B6B;
        --success: #4CAF50;
        --info: #2196F3;
        --code-bg: #ffffff;
        --btn-text: #ffffff;
        --border: rgba(0,0,0,0.06);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg: #0b0f13;
            --text: #e6eef6;
            --muted: #9aa4b2;
            --card-bg: #0f1418;
            --accent: #ff6b6b;
            --danger: #ff6b6b;
            --danger-variant: #ff8787;
            --success: #2ecc71;
            --info: #3399ff;
            --code-bg: #071014;
            --btn-text: #0b0f13;
            --border: rgba(255,255,255,0.06);
        }
    }

    .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .main {
        padding: 0rem 1rem;
    }

    .stButton>button {
        width: 100%;
        background-color: var(--danger) !important;
        color: var(--btn-text) !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid var(--border) !important;
    }
    .stButton>button:hover {
        background-color: var(--danger-variant) !important;
        transform: translateY(-1px);
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: var(--card-bg);
        margin: 1rem 0;
        color: var(--text);
        border: 1px solid var(--border);
    }

    .count-badge {
        display: inline-block;
        padding: 0.45rem 0.85rem;
        margin: 0.3rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .bus-badge { background-color: var(--danger); color: var(--btn-text); }
    .car-badge { background-color: var(--success); color: var(--btn-text); }
    .van-badge { background-color: var(--info); color: var(--btn-text); }

    .result-box code {
        font-size: 1.0rem;
        background-color: var(--code-bg);
        padding: 0.45rem;
        border-radius: 6px;
        display: inline-block;
        border: 1px solid var(--border);
        color: var(--text);
    }

    .footer-small {
        text-align: center;
        color: var(--muted);
        padding: 1rem;
    }

    .stImage > img {
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    try:
        model = YOLO('best_vehicle_detector.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def detect_vehicles(image, model, conf_threshold):
    if isinstance(image, Image.Image):
        image = np.array(image)
    results = model.predict(source=image, conf=conf_threshold, verbose=False)
    vehicle_counts = {'bus': 0, 'car': 0, 'van': 0}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name in vehicle_counts:
                vehicle_counts[class_name] += 1
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return vehicle_counts, annotated_image

def main():
    st.markdown("<h1 style='text-align: center; color: var(--accent);'>🚗 Vehicle Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: var(--muted);'>Deteksi dan hitung kendaraan (Bus, Car, Van) dalam gambar menggunakan YOLOv8s</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Nilai minimum confidence untuk deteksi objek"
        )
        st.markdown("---")
        st.markdown("### 📋 Tentang Aplikasi")
        st.info("""
        **Vehicle Detection System** menggunakan model YOLOv8s yang dilatih untuk mendeteksi:
        - 🚌 **Bus**
        - 🚗 **Car**
        - 🚐 **Van**
        """)

    with st.spinner("🔄 Loading model..."):
        model = load_model()
    if model is None:
        st.error("❌ Gagal memuat model.")
        return
    st.success("✅ Model berhasil dimuat!")

    st.markdown("### 📤 Upload Gambar")
    uploaded_file = st.file_uploader("Pilih gambar kendaraan...", type=['jpg', 'jpeg', 'png'], help="Upload gambar dalam format JPG, JPEG, atau PNG")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown(f"""
        <div class='result-box'>
            <b>📊 Info Gambar:</b><br>
            • Ukuran: {image.size[0]} x {image.size[1]} pixels<br>
            • Format: {image.format}<br>
            • Mode: {image.mode}
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.button("🚀 Mulai Deteksi", width='stretch'):
            with st.spinner("🔍 Mendeteksi kendaraan..."):
                vehicle_counts, annotated_image = detect_vehicles(image, model, conf_threshold)
                st.session_state.annotated_image = annotated_image
                st.session_state.vehicle_counts = vehicle_counts
                st.session_state.detection_ran = True

    if st.session_state.get('detection_ran', False):
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🖼️ Gambar Original")
            st.image(image, width='stretch')
        with col2:
            st.markdown("#### 🎯 Hasil Deteksi")
            st.image(st.session_state.annotated_image, width='stretch')
        vehicle_counts = st.session_state.vehicle_counts
        total_vehicles = sum(vehicle_counts.values())
        st.markdown(f"""
        <div class='result-box'>
            <h3>📊 Hasil Perhitungan</h3>
            <b>Total Kendaraan: {total_vehicles}</b><br>
            <span class='count-badge bus-badge'>🚌 Bus: {vehicle_counts['bus']}</span>
            <span class='count-badge car-badge'>🚗 Car: {vehicle_counts['car']}</span>
            <span class='count-badge van-badge'>🚐 Van: {vehicle_counts['van']}</span>
        </div>
        """, unsafe_allow_html=True)

        result_image = Image.fromarray(st.session_state.annotated_image)
        buf = io.BytesIO()
        result_image.save(buf, format='PNG')
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Hasil Deteksi",
            data=byte_im,
            file_name=f"detected_{uploaded_file.name}",
            mime="image/png",
            width='stretch'
        )

    st.markdown("---")
    st.markdown("### 🖼️ Contoh Penggunaan")
    st.markdown("""
    **Cara menggunakan aplikasi:**
    1. Upload gambar kendaraan melalui tombol upload
    2. Atur confidence threshold sesuai kebutuhan (default: 0.25)
    3. Klik tombol "Mulai Deteksi" untuk memproses gambar
    4. Lihat hasil deteksi dan jumlah kendaraan yang terdeteksi
    5. Download hasil deteksi jika diperlukan
    """)
    st.markdown("---")
    st.markdown("""
    <div class='footer-small'>
    <p><b>Vehicle Detection System</b> | Powered by YOLOv8s & Streamlit</p>
    <p>© 2025 - Dibuat untuk deteksi kendaraan otomatis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()