import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from tensorflow.keras.models import load_model

# ========================================
# Config
# ========================================

st.set_page_config(
    page_title="🌸 Flower Classification | CNN",
    page_icon="🌼",
    layout="centered"
)

IMAGE_SIZE = (180, 180)

MODEL_URL = "https://drive.google.com/uc?id=1D0chwazSgqzoDA0_d-Bx3XbQJnF0w6Ph"
CLASSES_URL = "https://drive.google.com/uc?id=1UI150DAPjVXHsGVZdvCTN2P0AsrFVHcE"

MODEL_PATH = "flower_cnn.model.h5"
CLASSES_PATH = "classes.npy"

FLOWER_EMOJI = {
    "daisy": "🌼",
    "dandelion": "🌾",
    "roses": "🌹",
    "sunflowers": "🌻",
    "tulips": "🌷"
}

# ========================================
# Helper Function
# ========================================

def download_file(url, path):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {path}..."):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

# ========================================
# Sidebar
# ========================================

st.sidebar.title("🌸 Flower Classification Project")

st.sidebar.markdown("""
### 📌 About
This project uses a **Convolutional Neural Network (CNN)**  
to classify flower images into 5 categories.

### 🛠 Tech Stack
- 🐍 Python  
- 🎈 Streamlit  
- 🧠 TensorFlow / Keras  
- 🔢 NumPy  
- 🖼 Pillow  
- 📂 OpenCV  

### 🌼 Classes
- Daisy  
- Dandelion  
- Roses  
- Sunflowers  
- Tulips  
""")

st.sidebar.markdown("---")
st.sidebar.info("Upload an image and let the model predict the flower 🌷")

# ========================================
# Load Model & Classes
# ========================================

download_file(MODEL_URL, MODEL_PATH)
download_file(CLASSES_URL, CLASSES_PATH)

@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH)
    return model, classes

model, classes = load_assets()

# ========================================
# Main UI
# ========================================

st.title("🌼 Flower Image Classification using CNN")
st.markdown(
    "Upload a flower image and the trained deep learning model will predict its category."
)

uploaded_file = st.file_uploader(
    "📤 Upload a Flower Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Flower"):
        with st.spinner("Analyzing image..."):
            image_resized = image.resize(IMAGE_SIZE)
            image_array = np.array(image_resized, dtype="float32") / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            preds = model.predict(image_array)
            class_index = np.argmax(preds)
            confidence = np.max(preds) * 100

            flower_name = classes[class_index]
            emoji = FLOWER_EMOJI.get(flower_name, "🌸")

        st.success(f"### {emoji} Prediction: **{flower_name.capitalize()}**")
        st.info(f"### 📊 Confidence: **{confidence:.2f}%**")

# ========================================
# Footer
# ========================================

st.markdown("""
---
<div style="text-align: center;">

Made with ❤️ by <b>Your Name</b>  

🔗 <a href="https://github.com/yourgithubusername" target="_blank">GitHub</a> | 
💼 <a href="https://www.linkedin.com/in/yourlinkedinusername" target="_blank">LinkedIn</a>

</div>
""", unsafe_allow_html=True)
