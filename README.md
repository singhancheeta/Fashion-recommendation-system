# 👗 Fashion Recommendation System

An AI-powered visual recommendation engine that suggests similar fashion items from a dataset of over 47,000 images using **deep learning** and **KNN-based similarity**.

---

## 🔍 What It Does

This system allows users to **upload an image of a clothing item** (like a dress, jacket, or shoe) and returns the **top 5 most visually similar items** using:

- 🎯 Deep CNN (ResNet18) for feature extraction  
- 🤝 Cosine similarity for finding visually close items  
- 🌐 Streamlit interface for easy web interaction

---

## 🧠 Key Features

- ✅ **ResNet18-based visual embeddings** using PyTorch  
- ✅ **K-Nearest Neighbors** (KNN) matching for top-5 results  
- ✅ **Streamlit-based web UI** for image upload and display  
- ✅ Works directly on folder-based datasets (e.g., `fashion_dataset/`)  
- ✅ Auto-computes embeddings on startup  

---

## 🧪 Project Workflow

```text
Fashion Image Dataset
     |
     |-- Resize & Normalize
     |-- Deep Feature Extraction (ResNet18)
     |-- Embedding Storage (in-memory NumPy)
     |
User Uploads Image
     |
     |-- Feature Extraction (ResNet18)
     |-- Cosine Similarity Matching
     |-- Return Top 5 Matches
```

---

## 🧰 Tech Stack

| Component         | Technology       |
|------------------|------------------|
| 👁 Image Features | PyTorch (ResNet18) |
| 📈 Similarity     | Scikit-learn (Cosine) |
| 🖼 Frontend       | Streamlit        |
| 🗃 Data Format    | Folder of Images |
| 💻 Language       | Python 3.10+     |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/singhancheeta/Fashion-recommendation-system.git
cd Fashion-recommendation-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** PyTorch and torchvision should match your system's CUDA or CPU version. You can install them manually if needed:  
> https://pytorch.org/get-started/locally/

### 3. Run the App

```bash
streamlit run app.py
```

---

## 🗂 Dataset Structure

```text
/fashion_dataset/
    10000.jpg
    10001.jpg
    10002.jpg
    ...
```

You can use the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) or any custom dataset with images.
,]

---

## 📦 To-Do / Enhancements

- [ ] Add labels/tags filtering (e.g., only shoes or jackets)
- [ ] Store precomputed features in `.npy` or MongoDB
- [ ] Add logging and better error handling
- [ ] Deploy on Streamlit Cloud or HuggingFace Spaces

---
