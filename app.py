import streamlit as st
from PIL import Image
from ultralytics import YOLO
import remedies  # your remedies.py file

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image or take a photo, then click 'Detect Disease' to get prediction and remedies.")

# ---------------------------
# Load YOLOv8 model
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your classification model

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
# Show image
# ---------------------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # ---------------------------
    # Detect Disease button
    # ---------------------------
    if st.button("Detect Disease"):
        with st.spinner("Predicting..."):
            results = model.predict(image, verbose=False)

            # ---------------------------
            # Safely get predicted class
            # ---------------------------
            try:
                # For classification models, ultralytics stores class predictions as:
                # results[0].boxes.cls (sometimes)
                # If not, results[0].names maps predicted index to class
                if hasattr(results[0], "probs") and results[0].probs is not None:
                    # Use probabilities to get class
                    pred_idx = int(results[0].probs.argmax())
                    pred_class = results[0].names[pred_idx]
                else:
                    # fallback: take the first predicted class from names
                    # works because ultralytics returns the predicted index
                    pred_class = list(results[0].names.values())[0]

            except Exception as e:
                st.error(f"Error extracting prediction: {e}")
                pred_class = None

            # ---------------------------
            # Display prediction and remedies
            # ---------------------------
            if pred_class:
                st.success(f"**Predicted Disease:** {pred_class}")

                if hasattr(remedies, pred_class):
                    st.markdown(f"### Remedies for {pred_class}:")
                    st.write(getattr(remedies, pred_class))
                else:
                    st.info("No remedies required or not found for this disease.")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.write("Classification model: YOLOv8 | Python 3.10.11 | No OpenCV required")
