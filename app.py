import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import remedies  # your remedies.py file

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image or take a photo to get disease prediction and remedies.")

# ---------------------------
# Load YOLOv8 Model
# ---------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # load your custom classification model
    return model

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
# Prediction
# ---------------------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Predicting..."):
        results = model.predict(image, verbose=False)
        pred_tensor = None

        # Try to get predictions safely
        if hasattr(results[0], "probs") and results[0].probs is not None:
            pred_tensor = results[0].probs
        elif hasattr(results[0], "x") and results[0].x is not None:
            pred_tensor = results[0].x  # raw logits from classification
        else:
            st.error("Cannot extract prediction from this model. Ensure it is a YOLOv8 classification model.")

        if pred_tensor is not None:
            # Top-3 predictions
            top_probs, top_idxs = torch.topk(pred_tensor, k=min(3, len(pred_tensor)))
            top_probs = top_probs.softmax(dim=0)  # convert logits to probabilities
            top_classes = [results[0].names[int(i)] for i in top_idxs]

            # Show top-1 prediction
            st.success(f"**Prediction:** {top_classes[0]} ({top_probs[0]*100:.2f}%)")

            # Show remedies for top-1
            if hasattr(remedies, top_classes[0]):
                st.markdown(f"### Remedies for {top_classes[0]}:")
                st.write(getattr(remedies, top_classes[0]))
            else:
                st.write("Remedy not found for this disease.")

            # Show top-3 predictions
            st.markdown("### Top 3 Predictions:")
            for cls, prob in zip(top_classes, top_probs):
                st.write(f"{cls}: {prob*100:.2f}%")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.write("Model trained with YOLOv8. Classification only, no bounding boxes.")
