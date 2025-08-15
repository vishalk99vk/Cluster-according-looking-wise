import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import io

# ---------------------------
# Load ResNet18 (lighter model)
# ---------------------------
@st.cache_resource
def load_model():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove final layer
    model.eval()
    return model, weights.transforms()

model, preprocess = load_model()

# ---------------------------
# Extract features from image
# ---------------------------
def get_features(image: Image.Image):
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0)
        features = model(img_tensor).squeeze().numpy()
    return features

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Image Similarity Grouper (Lightweight - ResNet18)")
st.write("Upload images and group them by visual similarity.")

uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = []
    filenames = []
    features = []

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        images.append(image)
        filenames.append(file.name)
        features.append(get_features(image))

    features = np.array(features).reshape(len(features), -1)

    n_clusters = st.slider("Number of clusters", 2, min(len(images), 10), 3)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(features)

    # Show grouped images
    for cluster_id in range(n_clusters):
        st.subheader(f"Cluster {cluster_id + 1}")
        cols = st.columns(5)
        idx = 0
        for img, name, label in zip(images, filenames, labels):
            if label == cluster_id:
                cols[idx % 5].image(img, caption=name, use_column_width=True)
                idx += 1
