import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import tempfile
from sklearn.metrics.pairwise import cosine_similarity

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
    return features / np.linalg.norm(features)  # Normalize for cosine similarity

# -------------------
# Group Images by 90% Similarity
# -------------------
def cluster_by_similarity(features, threshold=0.9):
    sim_matrix = cosine_similarity(features)
    visited = set()
    clusters = []
    
    for i in range(len(features)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(len(features)):
            if j not in visited and sim_matrix[i][j] >= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)
    return clusters

# -------------------
# Streamlit UI
# -------------------
st.title("🖼 Image Similarity Grouper (≥90% Similar)")

uploaded_files = st.file_uploader(
    "Upload multiple images",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files and st.button("🔍 Group Images"):
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
    st.write("Grouping by ≥90% similarity...")
    clusters = cluster_by_similarity(features, threshold=0.9)

    # Display results
    for idx, cluster in enumerate(clusters, start=1):
        st.subheader(f"Group {idx}")
        cols = st.columns(5)
        col_idx = 0
        for img_index in cluster:
            cols[col_idx].image(Image.open(file_paths[img_index]), use_column_width=True)
            col_idx = (col_idx + 1) % 5
