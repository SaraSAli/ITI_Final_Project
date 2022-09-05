import streamlit as st
from laneDetection import predict

st.header("Lane Detection Application")

image_path = st.text_input('Enter Image Path')

st.text(image_path)

if image_path:
    st.image(image_path)

submit = st.button('Submit')
if submit:
    predict(image_path)