import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import io

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
display_img_height = 120  # Smaller height for display
display_img_width = 120   # Smaller width for display

st.set_page_config(page_title="Fruit and Vegetable Classifier", page_icon="🍎", layout="wide")
st.title('Fruit and Vegetable Classification App')
st.write("Upload an image of a fruit or vegetable, and our model will classify it for you.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display loading spinner while processing
    with st.spinner('Processing your image...'):
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
            
            # Resize image for display
            image_for_display = image.resize((display_img_width, display_img_height))
            
            # Display results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image_for_display, caption='Uploaded Image', use_column_width=False, width=display_img_width)
                
            with col2:
                st.write(f'**Prediction:** {data_cat[class_idx]}')
                st.write(f'**Confidence:** {confidence:.2f}%')
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image file to classify.")

# Additional Information
st.sidebar.header("About")
st.sidebar.write("""
    This web app uses a TensorFlow model to classify fruits and vegetables from images.
    - **Model**: Trained on various fruit and vegetable images.
    - **Instructions**: Upload an image, and the app will display the predicted class and confidence.
""")
