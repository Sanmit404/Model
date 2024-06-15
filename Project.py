import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py as h
from PIL import Image
model = tf.keras.models.load_model("model.h5")
st.image("Untitled.jpg")
st.markdown('<center><h2>Check to see if your image is AI generated or Real</h2></center>', unsafe_allow_html=True)
st.markdown("""### """)
img = st.file_uploader("UPLOAD IMAGE HERE")
if st.button('CLICK TO CHECK'):
    pil_image = Image.open(img)
    resized_image = pil_image.resize((256, 256))
    rgb_array = np.array(resized_image.convert('RGB'))
    if(model.predict(np.expand_dims(rgb_array / 255, 0))>0.5):
        st.markdown("### Your image is _REAL_")
        st.image(img, width=705)
        mew = model.predict(np.expand_dims(rgb_array / 255, 0))
        mow = round(mew[0, 0]*100, 2)
        st.markdown(f"We can answer this with **{mow}**% centanity")
    else:
        st.markdown("### Your image is _AI generated_")
        st.image(img, width=705)
        mew = model.predict(np.expand_dims(rgb_array / 255, 0))
        mow = round((1-mew[0, 0])*100, 2)
        st.markdown(f"We can answer this with **{mow}**% centanity")
st.markdown("<h6 style='text-align: center; color: grey; font-size:13px'>Note: Our model works best with non-human images such as animals,art etc</h6>", unsafe_allow_html=True)
