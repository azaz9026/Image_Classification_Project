<<<<<<< HEAD

=======
>>>>>>> 4705dae252fc4505977c8c32b9e75d13adec5e4f
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os

# Load the model
model_path = 'Image_classify.keras'
model = load_model(model_path)

# Categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 
            'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 
            'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 
            'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
            'turnip', 'watermelon']

img_height = 180
img_width = 180

st.title('Fruit and Vegetable Classification')

st.write("Upload an image of a fruit or vegetable to get its classification.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display loading spinner while processing
    with st.spinner('Processing...'):
        image = Image.open(uploaded_file)
        image = image.resize((img_width, img_height))
        img_arr = tf.keras.preprocessing.image.img_to_array(image)
        img_bat = tf.expand_dims(img_arr, 0)
        
        # Predict
        try:
            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict)
            class_idx = np.argmax(score)
            confidence = np.max(score) * 100
            
            # Display image and prediction
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write(f'**Prediction:** {data_cat[class_idx]}')
            st.write(f'**Confidence:** {confidence:.2f}%')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write("Please upload an image file.")

