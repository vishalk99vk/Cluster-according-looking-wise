import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
import os
import tempfile

# -------------------
# Load Pre-trained Model (ResNet50)
# -------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()
    model = model.to(torch.device("cpu"))
    return model

model = load_model()

# -------------------
# Extract Features
# -------------------
def extract_features(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(img_t).squeeze().numpy()
    return features

# -------------------
# Streamlit UI
# -------------------
st.title("🖼 Image Similarity Grouper (PyTorch)")

eps_value = st.slider("Clustering sensitivity (eps)", 0.1, 5.0, 0.5, step=0.1)
min_samples_value = st.slider("Minimum images per group (min_samples)", 1, 10, 2)

uploaded_files = st.file_uploader(
    "Upload multiple images",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🔍 Re-cluster Images"):
        # Save uploaded files to temp folder
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        for file in uploaded_files:
            img_path = os.path.join(temp_dir, file.name)
            with open(img_path, "wb") as f:
                f.write(file.read())
            file_paths.append(img_path)

        # Extract features
        st.write("Extracting features...")
        features = [extract_features(Image.open(fp).convert("RGB")) for fp in file_paths]
        features = np.array(features)

        # Cluster images
        st.write("Clustering...")
        clustering = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric="euclidean").fit(features)
        labels = clustering.labels_

        # Display results
        unique_labels = set(labels)
        for label in sorted(unique_labels):
            st.subheader(f"Group {label}" if label != -1 else "Unclustered")
            cols = st.columns(5)
            col_idx = 0
            for idx, file_path in enumerate(file_paths):
                if labels[idx] == label:
                    cols[col_idx].image(Image.open(file_path), use_column_width=True)
                    col_idx = (col_idx + 1) % 5
