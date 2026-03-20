import streamlit as st
from PIL import Image
from ultralytics import YOLO
import remedies  # your remedies.py file

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image or take a photo, then click 'Detect Disease' to get prediction and remedies.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Image input
choice = st.radio("Select input method:", ["Upload Image", "Use Webcam"])
image = None
if choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png","webp"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif choice == "Use Webcam":
    webcam_image = st.camera_input("Take a picture of the leaf")
    if webcam_image:
        image = Image.open(webcam_image).convert("RGB")

if image:
    st.image(image, caption="Input Image", use_column_width=True)

    if st.button("Detect Disease"):
        with st.spinner("Predicting..."):
            results = model.predict(image, verbose=False)

            try:
                # Get top-1 prediction
                if hasattr(results[0], "probs") and results[0].probs is not None:
                    pred_idx = results[0].probs.top1
                    pred_class = results[0].names[pred_idx]
                    pred_conf = results[0].probs.top1conf.item()
                else:
                    pred_class = list(results[0].names.values())[0]
                    pred_conf = None
            except Exception as e:
                st.error(f"Error extracting prediction: {e}")
                pred_class = None
                pred_conf = None

            # Show prediction
            if pred_class:
                if pred_conf is not None:
                    st.success(f"**Predicted Disease:** {pred_class} ({pred_conf*100:.2f}%)")
                else:
                    st.success(f"**Predicted Disease:** {pred_class}")

                # ---------------------------
                # Lookup remedy in the dictionary
                # ---------------------------
                mapped_class = pred_class.lower().replace("___", "_").replace(" ", "_")

                if mapped_class in remedies.remedies:
                    st.markdown(f"### Remedies for {pred_class}:")
                    st.write(remedies.remedies[mapped_class])
                else:
                    st.info("No remedies required or not found for this disease.")

st.write("---")
st.write("Classification model: YOLOv8 | Python 3.10.11 | No OpenCV required")
