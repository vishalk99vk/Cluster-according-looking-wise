import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
from io import BytesIO

st.title("Image Similarity Clustering App")

# Cache model to avoid reloading
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

model = load_model()

uploaded_files = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Extract features from image (BytesIO or path)
def extract_features(img_file):
    img = Image.open(img_file).convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

# Extract common cluster name from filenames
def get_common_cluster_name(filenames):
    names = [fname.split('.')[0] for fname in filenames]
    if len(names) == 1:
        return names[0]
    split_names = [name.split() for name in names]
    common = set(split_names[0])
    for parts in split_names[1:]:
        common &= set(parts)
    common_name = " ".join([w for w in names[0].split() if w in common])
    return common_name if common_name else names[0]

if uploaded_files:
    st.info("Extracting features...")
    # Extract features for all images
    features = np.array([extract_features(file) for file in uploaded_files])

    st.info("Clustering images...")
    # DBSCAN clustering using cosine distance
    clustering = DBSCAN(eps=0.1, min_samples=1, metric='cosine').fit(features)
    labels = clustering.labels_

    clusters = {}
    for label, file in zip(labels, uploaded_files):
        clusters.setdefault(label, []).append(file)

    # Display clusters and prepare Excel
    data = []
    for cluster_id, files in clusters.items():
        cluster_filenames = [f.name for f in files]
        cluster_name = get_common_cluster_name(cluster_filenames)
        st.subheader(f"Cluster: {cluster_name}")
        cols = st.columns(len(files))
        for col, f in zip(cols, files):
            img = Image.open(f)
            col.image(img, caption=f.name, use_container_width=True)
            name_no_ext = f.name.split('.')[0]
            data.append([cluster_name, name_no_ext, f.name])

    df = pd.DataFrame(data, columns=["Cluster Name", "Image Name (No Ext)", "Exact Filename"])

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
