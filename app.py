import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests 
from PIL import Image
import numpy as np
from scipy import ndimage

st.title("Draw a Digit")
import os
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

canvas=st_canvas(height=280,width=280, stroke_width=10 ,stroke_color="#ffffff", background_color="#000000")

def center_digit(array):
        # array: your grayscale 280x280 array, BEFORE the /255 normalization step
    
        # 1. Bounding box (what already reasoned through)
        rows, cols = np.where(array > 50)
        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()
        cropped = array[top:bottom+1, left:right+1]

        # 2. Resize so the LONGER side becomes 20px, preserving aspect ratio
        h, w = cropped.shape
        if h > w:
            new_h, new_w = 20, max(1, int(round(w * 20 / h)))
        else:
            new_w, new_h = 20, max(1, int(round(h * 20 / w)))
        resized = np.array(Image.fromarray(cropped).resize((new_w, new_h)))

        # 3. Paste onto a blank 28x28 canvas, geometrically centered first
        canvas28 = np.zeros((28, 28))
        pad_top = (28 - new_h) // 2
        pad_left = (28 - new_w) // 2
        canvas28[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        # 4. Shift so the CENTER OF MASS lands at (14, 14), not just geometric center
        cy, cx = ndimage.center_of_mass(canvas28)
        shift_y, shift_x = int(round(14 - cy)), int(round(14 - cx))
        canvas28 = ndimage.shift(canvas28, (shift_y, shift_x))
        return canvas28

if st.button("Predict"):
    img = Image.fromarray(canvas.image_data)
    img = img.convert("L")
    array = np.array(img)
    array = center_digit(array)
    array = array / 255
    array = (array - 0.5) / 0.5
    array = array.flatten()
    pixel_values = array.tolist()

    response = requests.post(BACKEND_URL+"/predict", json={"pixel_values": pixel_values})
    data = response.json()
    predicted_class = data.get("Predicted_Class")
    confidence = data.get("Prob")
    st.write("Predicted Digit:", predicted_class)
    st.write("Confidence:", confidence)


    all_probs = data.get("All_Probs")
    st.bar_chart(all_probs)
