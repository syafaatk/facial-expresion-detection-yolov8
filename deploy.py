import av
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO

st.title("Deteksi Ekspresi Wajah")

model_options = {
    "Best yolov8n (hasil training)": "runs/detect/train/weights/best.pt",
    "Last yolov8n (hasil training)": "runs/detect/train/weights/last.pt",
    "Best yolov8x (hasil training)": "runs/detect/train/weights/yolov8x/best_yolov8x.pt",
}

EMOTION_NAMES = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neutral",
    6: "Sad",
    7: "Surprise",
}

@st.cache_resource
def load_model(path):
    model = YOLO(path)
    if len(model.names) == 8:
        model.names = EMOTION_NAMES
    return model

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
    conf = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.25)

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=conf, verbose=False)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="deteksi-live",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )