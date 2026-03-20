import streamlit as st
from PIL import Image
from ultralytics import YOLO
import remedies  # your remedies.py file

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image or take a photo, then click 'Detect Disease' to get prediction and remedies.")

# ---------------------------
# Load YOLOv8 Model
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # load your custom classification model

model = load_model()

# ---------------------------
# Image Input
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
    # Detect Disease Button
    # ---------------------------
    if st.button("Detect Disease"):
        with st.spinner("Predicting..."):
            results = model.predict(image, verbose=False)

            # Safe extraction of predicted class
            try:
                pred_class = list(results[0].names.values())[0]
            except Exception:
                st.error("Cannot extract prediction from this model.")
                pred_class = None

            if pred_class:
                st.success(f"**Prediction:** {pred_class}")

                # Remedies
                if hasattr(remedies, pred_class):
                    st.markdown(f"### Remedies for {pred_class}:")
                    st.write(getattr(remedies, pred_class))
                else:
                    st.write("Remedy not found for this disease.")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.write("Model trained with YOLOv8. Classification only, no bounding boxes or probabilities.")
