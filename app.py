import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import pandas as pd
from io import BytesIO

st.title("Image Similarity Clustering App")

# Session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded:
    st.session_state.uploaded_files.extend(uploaded)

if st.button("🗑️ Clear All"):
    st.session_state.uploaded_files = []
    st.stop()


def extract_features(img_file):
    """Extract features from an image using resized pixels and color histogram."""
    img = Image.open(img_file).convert("RGB").resize((64, 64))
    img_array = np.array(img).flatten() / 255.0  # normalized pixel values
    # Simple RGB histogram
    hist = np.histogram(img_array, bins=64)[0]
    return np.concatenate([img_array, hist])


def get_common_cluster_name(filenames):
    names = [os.path.splitext(f)[0] for f in filenames]
    if not names:
        return "Unknown Cluster"
    if len(names) == 1:
        return names[0]
    split_names = [name.split() for name in names]
    common = set(split_names[0])
    for parts in split_names[1:]:
        common &= set(parts)
    common_name = " ".join([w for w in names[0].split() if w in common])
    return common_name if common_name else names[0]


if st.session_state.uploaded_files:
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in st.session_state.uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path)

    features = np.array([extract_features(path) for path in file_paths])

    sim_matrix = cosine_similarity(features)
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    dist_matrix = 1 - sim_matrix

    dbscan = DBSCAN(eps=0.5, min_samples=1, metric="precomputed")
    labels = dbscan.fit_predict(dist_matrix)

    clusters_dict = {}
    for i, label in enumerate(labels):
        clusters_dict.setdefault(label, []).append(i)
    clusters = list(clusters_dict.values())

    # Create DataFrame for Excel
    data = []
    for cluster in clusters:
        cluster_files = [os.path.basename(file_paths[i]) for i in cluster]
        cluster_name = get_common_cluster_name(cluster_files)
        for fname in cluster_files:
            data.append([cluster_name, os.path.splitext(fname)[0], fname])
    df = pd.DataFrame(data, columns=["Cluster Name", "Image Name (No Ext)", "Exact Filename"])

    # Display clusters
    if not clusters:
        st.info("No clusters found.")
    else:
        for cluster in clusters:
            cluster_files = [os.path.basename(file_paths[i]) for i in cluster]
            cluster_name = get_common_cluster_name(cluster_files)
            st.subheader(f"Cluster: {cluster_name}")
            cols = st.columns(len(cluster_files))
            for col, idx in zip(cols, cluster):
                img = Image.open(file_paths[idx])
                col.image(img, caption=os.path.basename(file_paths[idx]), use_container_width=True)

    # Excel download
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Download Clusters Excel",
        data=excel_buffer,
        file_name="image_clusters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
