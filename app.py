import streamlit as st
import tempfile
from PIL import Image
from recommender import FashionRecommender

DATASET_PATH = "fashion_dataset" 
recommender = FashionRecommender(DATASET_PATH)

st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("👗 Fashion Recommendation System with ResNet (PyTorch)")

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        query_image_path = tmp.name

    st.image(Image.open(query_image_path), caption="Query Image", width=250)
    st.write("🔍 Finding similar images...")

    recommendations = recommender.recommend(query_image_path)

    st.subheader("🧠 Top 5 Recommendations")
    cols = st.columns(5)
    for i, rec in enumerate(recommendations):
        with cols[i]:
            st.image(rec, use_container_width=True, caption=f"Match {i+1}")
