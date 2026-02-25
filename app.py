import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from tensorflow.keras.models import load_model

IMAGE_SIZE = (180,180)

MODEL_URL = 'https://drive.google.com/uc?id=1D0chwazSgqzoDA0_d-Bx3XbQJnF0w6Ph'
CLASSES_URL = 'https://drive.google.com/uc?id=1UI150DAPjVXHsGVZdvCTN2P0AsrFVHcE'

MODEL_PATH = 'flower_cnn.model.h5'
CLASSES_PATH = 'clases.npy'


FLOWER_EMOJI = {
    "daisy": "🌼",
    "dandelion": "🌻",
    "roses": "🌹",
    "sunflowers": "🌻",
    "tulips": "🌷"
}

def download_file(url,path):
    if not os.path.exists(path):
        r = requests.get(url, stream = True)
        r.raise_for_status()

        with open(path,'wb') as f:
            for chunk in r.iter_content(chunk_size = 8192):
                if chunk:
                    f.write(chunk)




# Download important Assests

download_file(MODEL_URL,MODEL_PATH)
download_file(CLASSES_URL,CLASSES_PATH)

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)



# ========================================
# Build Web-app

st.set_page_config(
    page_title = 'Flower Classification | CNN',
    page_icon = '🌼',
    layout = 'centered'
)


st.sidebar.title('Flower Classification Project')
st.sidebar.markdown('I Will Write More About this')

uploaded_file = st.file_uploader(
    '🌼 Upload a Flower Image ',
    type = ['jpg','jpeg','png','webp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption = 'Uploaded Image', width = 200)

    if st.button('Predict Flower: '):
        with st.spinner('Analyzing image...'):
            image_resized = image.resize(IMAGE_SIZE)
            image_array = np.array(image_resized, dtype = 'float32') / 255.0
            image_array = np.expand_dims(image_array,axis = 0)

            preds = model.predict(image_array)
            class_index  = np.argmax(preds)
            confindence = np.max(preds)*100

            flower_name = classes[class_index]
            emoji = FLOWER_EMOJI.get(flower_name, "🌸")

        st.success(f'### {emoji} Prediction : **{flower_name.capitalize()}**')
        st.info(f'### 📊 Confindence: **{confindence:.2f}**')


st.markdown("""
---
<div style="text-align: center;">

Made with ❤️ by <b>Your Name</b>  

🔗 <a href="https://github.com/yourgithubusername" target="_blank">GitHub</a> | 
💼 <a href="https://www.linkedin.com/in/yourlinkedinusername" target="_blank">LinkedIn</a>

</div>
""", unsafe_allow_html=True)













                    
