import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py as h
from PIL import Image
model = tf.keras.models.load_model("model.h5")
st.image("Untitled.jpg")
st.markdown("""### Check if your image is AI generated or Real 
## 
""")
img = st.file_uploader("UPLOAD IMAGE HERE")
if st.button('CHECK'):
    pil_image = Image.open(img)
    resized_image = pil_image.resize((256, 256))
    rgb_array = np.array(resized_image.convert('RGB'))
    if(model.predict(np.expand_dims(rgb_array / 255, 0))>0.5):
        st.markdown("### Your image is _REAL_")
        st.image(img,width=500)
        mew = model.predict(np.expand_dims(rgb_array / 255, 0))
        mow = mew[0, 0]
        st.markdown(f"We can answer this with **{mow}**% centanity")
    else:
        st.markdown("### Your image is _AI generated_")
        st.image(img,width=500)
        mew = model.predict(np.expand_dims(rgb_array / 255, 0))
        mow = 1-mew[0, 0]
        st.markdown(f"We can answer this with **{mow}**% centanity")

