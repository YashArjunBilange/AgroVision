# app.py
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import remedies  # your remedies.py

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image or take a photo to get disease prediction and remedies.")

# ---------------------------
# Load YOLO model
# ---------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # load your custom classification model
    return model

model = load_model()

# ---------------------------
# Image input
# ---------------------------
choice = st.radio("Select input method:", ["Upload Image", "Use Webcam"])

image = None
if choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif choice == "Use Webcam":
    webcam_image = st.camera_input("Take a picture of the leaf")
    if webcam_image:
        image = Image.open(webcam_image).convert("RGB")

# ---------------------------
# Prediction
# ---------------------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Predicting..."):
        results = model.predict(image)
        try:
            class_idx = int(results[0].probs.argmax())  # top probability
            prediction = results[0].names[class_idx]
        except AttributeError:
            # fallback if probs not available
            prediction = results[0].names[int(results[0].boxes.cls[0])]
            
        st.success(f"**Prediction:** {prediction}")

        # Remedies
        if hasattr(remedies, prediction):
            st.markdown(f"### Remedies for {prediction}:")
            st.write(getattr(remedies, prediction))
        else:
            st.write("Remedy not found for this disease.")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.write("Model trained with YOLOv8. Classification only, no bounding boxes.")
