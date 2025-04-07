import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("trained.h5")

# Set title
st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a Chest X-Ray image to check for **Pneumonia**.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img = np.array(image)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    if img.ndim == 2:  # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_resized = cv2.resize(img, (300, 300))

    img_scaled = img_resized / 255.0

    img_reshaped = img_scaled.reshape(1, 300, 300, 3)

    prediction = model.predict(img_reshaped)[0][0]
    label = "Pneumonia" if prediction >= 0.5 else "Normal"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence:.2f}")

    if label == "Pneumonia":
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Exclamation_mark_in_triangle.svg/1024px-Exclamation_mark_in_triangle.svg.png", caption="Pneumonia Detected", use_container_width=True)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Check_mark_icon.svg/1024px-Check_mark_icon.svg.png", caption="Normal", use_container_width=True)
