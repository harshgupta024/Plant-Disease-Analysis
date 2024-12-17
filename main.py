import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load the pre-trained model and class indices
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.set_page_config(page_title="Plant Disease Classifier", page_icon=":seedling:", layout="wide")

# Main layout
st.title("🌿 Plant Disease Classifier")
st.write("Upload an image of a plant leaf to classify its disease.")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Uploaded Image")
        resized_img = image.resize((300, 300))
        st.image(resized_img)
    
    with col2:
        st.header("Prediction")
        if st.button("Classify"):
            with st.spinner("Classifying..."):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"Prediction: **{str(prediction)}**")
                st.balloons()
else:
    st.warning("Please upload an image to get started.")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    """,
    unsafe_allow_html=True
)
