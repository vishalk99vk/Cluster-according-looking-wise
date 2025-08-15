import streamlit as st
import os
import tempfile
import shutil
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from io import BytesIO
import openpyxl

# ========== IMAGE FEATURE EXTRACTION ==========
def get_image_feature_vector(img_path):
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    arr = np.array(img) / 255.0
    return arr.flatten()

def get_common_name(names):
    if not names:
        return "Unknown"
    split_names = [n.split('-') for n in names]
    min_parts = min(len(s) for s in split_names)
    common_parts = []
    for i in range(min_parts):
        part_set = set(s[i].strip() for s in split_names)
        if len(part_set) == 1:
            common_parts.append(part_set.pop())
        else:
            break
    return ' - '.join(common_parts) if common_parts else names[0]

# ========== CLUSTERING BY SIMILARITY ==========
def cluster_images(image_paths, similarity_threshold=0.9):
    features = [get_image_feature_vector(p) for p in image_paths]
    similarity_matrix = cosine_similarity(features)

    clusters = []
    visited = set()
    for i, path in enumerate(image_paths):
        if i in visited:
            continue
        cluster = [path]
        visited.add(i)
        for j in range(i+1, len(image_paths)):
            if j not in visited and similarity_matrix[i][j] >= similarity_threshold:
                cluster.append(image_paths[j])
                visited.add(j)
        clusters.append(cluster)
    return clusters

# ========== EXCEL EXPORT ==========
def export_clusters_to_excel(clusters):
    data = []
    for cluster in clusters:
        cluster_names = [os.path.splitext(os.path.basename(p))[0] for p in cluster]
        full_names = [os.path.basename(p) for p in cluster]
        cluster_name = get_common_name(cluster_names)
        for cn, fn in zip(cluster_names, full_names):
            data.append([cluster_name, cn, fn])
    df = pd.DataFrame(data, columns=["Cluster Name", "Image Name (No Ext)", "Full Image Name"])
    
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return output

# ========== STREAMLIT UI ==========
st.title("📸 Image Similarity Clustering (90%) with Disk Fallback")

uploaded_files = st.file_uploader(
    "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    total_size = sum([len(f.getbuffer()) for f in uploaded_files]) / (1024 * 1024)
    st.write(f"Total upload size: **{total_size:.2f} MB**")

    if total_size <= 150:
        # In-memory processing
        st.info("Processing in memory...")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for file in uploaded_files:
                file_path = os.path.join(tmpdir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                paths.append(file_path)

            clusters = cluster_images(paths, similarity_threshold=0.9)

    else:
        # Disk processing for heavy uploads
        st.warning("Large upload detected. Using disk processing to avoid memory issues...")
        temp_folder = "temp_processing"
        os.makedirs(temp_folder, exist_ok=True)

        paths = []
        for file in uploaded_files:
            file_path = os.path.join(temp_folder, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            paths.append(file_path)

        clusters = cluster_images(paths, similarity_threshold=0.9)

        # Cleanup after processing
        shutil.rmtree(temp_folder)

    # Display clusters
    for idx, cluster in enumerate(clusters, 1):
        cluster_names = [os.path.splitext(os.path.basename(p))[0] for p in cluster]
        cluster_display_name = get_common_name(cluster_names)
        st.subheader(f"🗂 Cluster: {cluster_display_name}")
        cols = st.columns(5)
        for i, img_path in enumerate(cluster):
            with cols[i % 5]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

    # Export to Excel
    excel_data = export_clusters_to_excel(clusters)
    st.download_button(
        label="📥 Download Cluster Data (Excel)",
        data=excel_data,
        file_name="image_clusters.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
