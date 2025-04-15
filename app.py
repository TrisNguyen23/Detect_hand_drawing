import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import cv2

st.set_page_config(page_title="AI Vẽ Tay", page_icon="✏️")

st.title("AI Nhận Diện Hình Vẽ Tay")
st.write("Vẽ một hình đơn giản rồi nhấn **Dự đoán**!")

model = tf.keras.models.load_model("./model.h5")
class_names = ['cat', 'dog', 'crocodile', 'elephant','cow','duck','crab','dragon']

canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Dự đoán"):
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0]  
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)
        class_idx = np.argmax(pred)
        confidence = np.max(pred)

        st.success(f"AI đoán: **{class_names[class_idx]}** ({confidence*100:.2f}%)")
    else:
        st.warning("Bạn chưa vẽ gì cả!")
