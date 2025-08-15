import os
import shutil
import numpy as np
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from difflib import SequenceMatcher
from tensorflow.keras.models import Model
from PIL import Image
import tempfile

# Load pre-trained ResNet50 model without top layer
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Extract deep features from image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

# Find longest common substring from filenames
def get_common_name(file_list):
    if not file_list:
        return "Unknown"
    names = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    common = names[0]
    for name in names[1:]:
        match = SequenceMatcher(None, common, name).find_longest_match(0, len(common), 0, len(name))
        common = common[match.a: match.a + match.size]
    common = common.strip(" -_()")
    return common if common else names[0]

# Main clustering function
def cluster_images(img_paths, similarity_threshold=0.9):
    features = [extract_features(p) for p in img_paths]
    sim_matrix = cosine_similarity(features)

    visited = set()
    clusters = []
    for i in range(len(img_paths)):
        if i in visited:
            continue
        cluster = [img_paths[i]]
        visited.add(i)
        for j in range(i + 1, len(img_paths)):
            if j not in visited and sim_matrix[i][j] >= similarity_threshold:
                cluster.append(img_paths[j])
                visited.add(j)
        clusters.append(cluster)
    return clusters

# Streamlit app
st.title("Image Similarity Clustering (ResNet50)")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        img_paths = []
        for file in uploaded_files:
            img_path = os.path.join(temp_dir, file.name)
            with open(img_path, "wb") as f:
                f.write(file.read())
            img_paths.append(img_path)

        clusters = cluster_images(img_paths, similarity_threshold=0.9)

        # Save and display clusters
        output_dir = os.path.join(temp_dir, "clusters")
        os.makedirs(output_dir, exist_ok=True)

        for cluster in clusters:
            cluster_name = get_common_name(cluster)
            cluster_folder = os.path.join(output_dir, cluster_name)
            os.makedirs(cluster_folder, exist_ok=True)

            st.subheader(f"Cluster: {cluster_name}")
            cols = st.columns(len(cluster))

            for idx, img_path in enumerate(cluster):
                shutil.copy(img_path, cluster_folder)
                with cols[idx]:
                    st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)

        # Zip clusters for download
        zip_path = shutil.make_archive(os.path.join(temp_dir, "clusters"), 'zip', output_dir)
        with open(zip_path, "rb") as f:
            st.download_button("Download All Clusters (ZIP)", f, "clusters.zip", "application/zip")
