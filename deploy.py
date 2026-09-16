import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Deteksi Ekspresi Wajah")

model_options = {
    "Best (hasil training)": "runs/detect/train/weights/best.pt",
    "Last (hasil training)": "runs/detect/train/weights/last.pt",
    "YOLOv8n": "yolov8n.pt",
    "YOLO11n": "yolo11n.pt",
}

@st.cache_resource
def load_model(path):
    return YOLO(path)

model_name = st.sidebar.selectbox("Pilih Model", list(model_options.keys()))
model = load_model(model_options[model_name])

input_type = st.sidebar.radio("Tipe Input", ["File", "Kamera"])

if input_type == "File":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model(img)
        st.image(results[0].plot(), caption="Hasil Deteksi")
else:
    camera_file = st.camera_input("Ambil Foto dari Kamera")
    if camera_file:
        img = Image.open(camera_file)
        results = model(img)
        st.image(results[0].plot(), caption="Hasil Deteksi")