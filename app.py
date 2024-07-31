

import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import os

# Load the model
model_path = os.path.join('C:', 'Users', 'mdaza', 'OneDrive', 'Desktop', 'deep folder', 'Image_classify.keras')
model = load_model(r'C:\Users\mdaza\OneDrive\Desktop\deep folder\Image_classify.keras')

# Categories
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 
            'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 
            'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 
            'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
            'turnip', 'watermelon']

img_height = 180
img_width = 180

st.header('Image Classification')

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((img_height, img_width))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)
    
    # Predict
    try:
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
        st.write('with the accuracy of ' + str(np.max(score) * 100) + '%')
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write("Please upload an image file.")
