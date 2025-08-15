import streamlit as st
import os
import tempfile
import shutil
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openpyxl import Workbook

# ---------------------------
# UTILS
# ---------------------------
def image_to_feature_vector(image_path, size=(64, 64)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    return np.array(img).flatten()

def cluster_images(image_paths, similarity_threshold=0.9):
    features = [image_to_feature_vector(p) for p in image_paths]
    features = np.array(features)
    sim_matrix = cosine_similarity(features)

    clusters = []
    visited = set()

    for i in range(len(image_paths)):
        if i in visited:
            continue
        cluster = [image_paths[i]]
        visited.add(i)
        for j in range(i+1, len(image_paths)):
            if j not in visited and sim_matrix[i, j] >= similarity_threshold:
                cluster.append(image_paths[j])
                visited.add(j)
        clusters.append(cluster)
    return clusters

def get_cluster_name_from_files(file_names):
    # Extract words from first file and find common words with others
    first_parts = file_names[0].split()
    common_parts = set(first_parts)
    for name in file_names[1:]:
        common_parts &= set(name.split())
    if common_parts:
        return " ".join(common_parts)
    return os.path.splitext(file_names[0])[0]

def save_clusters_to_excel(clusters, output_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Cluster Name", "Image Name (no ext)", "Exact File Name"])

    for cluster in clusters:
        file_names = [os.path.basename(p) for p in cluster]
        cluster_name = get_cluster_name_from_files(file_names)
        for fname in file_names:
            ws.append([
                cluster_name,
                os.path.splitext(fname)[0],
                fname
            ])
    wb.save(output_path)

# ---------------------------
# STREAMLIT APP
# ---------------------------
st.title("📸 Image Clustering App (90% Similarity)")
st.write("Upload images — we'll group them by **visual similarity** and name clusters from SKU names.")

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    total_size = sum([len(f.getbuffer()) for f in uploaded_files])
    st.write(f"Total upload size: {total_size / (1024*1024):.2f} MB")

    # Decide processing mode
    if total_size > 150 * 1024 * 1024:
        st.warning("⚠️ Large upload — processing locally to avoid memory crash.")
        tmpdir = tempfile.mkdtemp()
        image_paths = []
        for file in uploaded_files:
            path = os.path.join(tmpdir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(path)
    else:
        tmpdir = tempfile.mkdtemp()
        image_paths = []
        for file in uploaded_files:
            path = os.path.join(tmpdir, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(path)

    with st.spinner("🔍 Clustering images..."):
        clusters = cluster_images(image_paths, similarity_threshold=0.9)

    # Show clusters
    for idx, cluster in enumerate(clusters, 1):
        file_names = [os.path.basename(p) for p in cluster]
        cluster_name = get_cluster_name_from_files(file_names)
        st.subheader(f"🗂 {cluster_name}")
        cols = st.columns(5)
        for i, img_path in enumerate(cluster):
            img = Image.open(img_path)
            with cols[i % 5]:
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)

    # Save to Excel
    excel_path = os.path.join(tmpdir, "clusters.xlsx")
    save_clusters_to_excel(clusters, excel_path)

    with open(excel_path, "rb") as f:
        st.download_button(
            label="📥 Download Clusters Excel",
            data=f,
            file_name="clusters.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Cleanup will happen automatically when app restarts
