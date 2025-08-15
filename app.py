import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
from io import BytesIO

# Load ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

st.title("Image Similarity Clustering App")

uploaded_files = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def get_common_cluster_name(filenames):
    # Extract common substring ignoring numbers and extensions
    names = [os.path.splitext(f)[0] for f in filenames]
    if len(names) == 1:
        return names[0]
    split_names = [name.split() for name in names]
    common = set(split_names[0])
    for parts in split_names[1:]:
        common &= set(parts)
    common_name = " ".join([w for w in names[0].split() if w in common])
    return common_name if common_name else names[0]

if uploaded_files:
    # Save uploaded files temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path)

    # Extract features
    features = [extract_features(path) for path in file_paths]
    features = np.array(features)

    # Compute similarity
    sim_matrix = cosine_similarity(features)

    # Clustering (manual based on threshold)
    threshold = 0.9
    visited = set()
    clusters = []
    for idx, file in enumerate(file_paths):
        if idx in visited:
            continue
        cluster = [idx]
        visited.add(idx)
        for j in range(idx + 1, len(file_paths)):
            if j not in visited and sim_matrix[idx, j] >= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)

    # Prepare DataFrame for Excel
    data = []
    for cluster in clusters:
        cluster_files = [os.path.basename(file_paths[i]) for i in cluster]
        cluster_name = get_common_cluster_name(cluster_files)
        for fname in cluster_files:
            name_no_ext = os.path.splitext(fname)[0]
            data.append([cluster_name, name_no_ext, fname])

    df = pd.DataFrame(data, columns=["Cluster Name", "Image Name (No Ext)", "Exact Filename"])

    # Display clusters in Streamlit
    for cluster in clusters:
        cluster_files = [os.path.basename(file_paths[i]) for i in cluster]
        cluster_name = get_common_cluster_name(cluster_files)
        st.subheader(f"Cluster: {cluster_name}")
        cols = st.columns(len(cluster_files))
        for col, idx in zip(cols, cluster):
            img = Image.open(file_paths[idx])
            col.image(img, caption=os.path.basename(file_paths[idx]), use_container_width=True)

    # Download Excel
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Download Clusters Excel",
        data=excel_buffer,
        file_name="image_clusters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
