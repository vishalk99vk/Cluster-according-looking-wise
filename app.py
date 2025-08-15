import os
import shutil
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model
@st.cache_resource
def load_model():
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

model = load_model()

# Extract features
def get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

# Main Streamlit app
st.title("📂 Image Similarity Clustering (90%+ match)")
st.write("Uploads images — groups them if they're at least 90% similar (ignores size).")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    # Save uploaded files to temp dir
    temp_dir = "uploaded_images"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        file_paths.append(path)

    # Extract features
    st.write("🔍 Extracting features...")
    features = [get_features(p) for p in file_paths]

    # Clustering manually (90%+ similarity)
    clusters = []
    visited = set()
    for i, feat1 in enumerate(features):
        if i in visited:
            continue
        cluster = [file_paths[i]]
        visited.add(i)
        for j, feat2 in enumerate(features):
            if j not in visited:
                sim = cosine_similarity([feat1], [feat2])[0][0]
                if sim >= 0.90:
                    cluster.append(file_paths[j])
                    visited.add(j)
        clusters.append(cluster)

    # Save clusters into folders
    output_dir = "clusters"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for idx, cluster in enumerate(clusters, start=1):
        cluster_folder = os.path.join(output_dir, f"cluster_{idx}")
        os.makedirs(cluster_folder, exist_ok=True)
        for img_path in cluster:
            shutil.copy(img_path, cluster_folder)

    st.success(f"✅ Found {len(clusters)} clusters. Saved in '{output_dir}' folder.")
    for idx, cluster in enumerate(clusters, start=1):
        st.subheader(f"Cluster {idx}")
        for img_path in cluster:
            st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
