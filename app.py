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
# Load YOLOv8 classification model
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------------------
# Image input
# ---------------------------
choice = st.radio("Select input method:", ["Upload Image", "Use Webcam"])
image = None
if choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif choice == "Use Webcam":
    webcam_image = st.camera_input("Take a picture of the leaf")
    if webcam_image:
        image = Image.open(webcam_image).convert("RGB")

# ---------------------------
# Display image
# ---------------------------
if image:
    st.image(image, caption="Input Image", use_column_width=True)

    # ---------------------------
    # Detect Disease button
    # ---------------------------
    if st.button("Detect Disease"):
        with st.spinner("Predicting..."):
            results = model.predict(image, verbose=False)

            try:
                # Get top-3 predictions from Probs object
                if hasattr(results[0], "probs") and results[0].probs is not None:
                    top_indices = results[0].probs.top5[:3]  # top-3 indices
                    top_confs = results[0].probs.top5conf[:3]  # top-3 confidences
                    top_classes = [results[0].names[i] for i in top_indices]
                else:
                    # fallback for older versions
                    top_classes = list(results[0].names.values())[:3]
                    top_confs = [None]*len(top_classes)
            except Exception as e:
                st.error(f"Error extracting prediction: {e}")
                top_classes = []
                top_confs = []

            # ---------------------------
            # Show predictions and remedies
            # ---------------------------
            for cls, conf in zip(top_classes, top_confs):
                if conf is not None:
                    st.success(f"**Predicted Disease:** {cls} ({conf*100:.2f}%)")
                else:
                    st.success(f"**Predicted Disease:** {cls}")

                # Lookup remedy in dictionary
                mapped_class = cls.lower().replace("___", "_").replace(" ", "_")
                if mapped_class in remedies.remedies:
                    st.markdown(f"### Remedies for {cls}:")
                    st.write(remedies.remedies[mapped_class])
                else:
                    st.info("No remedies required or not found for this disease.")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.write("Classification model: YOLOv8 | Python 3.10.11 | No OpenCV required")
