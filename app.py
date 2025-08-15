import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# ---------------------------
# Load CLIP Model
# ---------------------------
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_clip_model()

# ---------------------------
# Extract features
# ---------------------------
def get_features(image: Image.Image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Image Similarity Grouper (CLIP-powered)")
st.write("Upload images — they’ll be grouped by visual & semantic similarity.")

uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = []
    filenames = []
    features = []

    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        images.append(img)
        filenames.append(file.name)
        features.append(get_features(img))

    features = np.array(features)

    n_clusters = st.slider("Number of clusters", 2, min(len(images), 10), 3)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(features)

    # Show results
    for cluster_id in range(n_clusters):
        st.subheader(f"Cluster {cluster_id + 1}")
        cols = st.columns(5)
        idx = 0
        for img, name, label in zip(images, filenames, labels):
            if label == cluster_id:
                cols[idx % 5].image(img, caption=name, use_column_width=True)
                idx += 1
