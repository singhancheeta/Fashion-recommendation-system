import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st

st.header('Fashion Recommendation System')

# Load pre-computed image features and filenames
Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Define a function to extract features from images
def extract_feature_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Create NearestNeighbors instance
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)  # Fit the NearestNeighbors model with pre-computed image features

# File upload widget
upload_file = st.file_uploader("Upload Image")
if upload_file is not None:
    # Save uploaded image to disk
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    
    # Display uploaded image
    st.subheader('Uploaded Image')
    st.image(upload_file, width=250)

    # Extract features from uploaded image
    input_img_features = extract_feature_from_images(upload_file, model)

    # Find nearest neighbors
    distances, indices = neighbors.kneighbors([input_img_features])

    # Display recommended images
    st.subheader('Recommended Images')
    cols = st.columns(5)
    for i, col in enumerate(cols):
        col.image(filenames[indices[0][i+1]])
