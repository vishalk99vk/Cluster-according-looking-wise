import os
import shutil
import numpy as np
import streamlit as st
from sklearn.cluster import DBSCAN
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image

# ==== CONFIG ====
resize_dim = (224, 224)              # ResNet50 input size
eps_value = 0.5                      # DBSCAN sensitivity
min_samples_value = 2                # Minimum images per group
cache_file = "image_features.npy"    # Cache file for embeddings

# ==== Load Model ====
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)

model = load_model()

# ==== Extract Deep Features ====
def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=resize_dim)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = model.predict(img_data, verbose=0)
        return features.flatten()
    except:
        return None

# ==== Streamlit UI ====
st.title("🖼 Image Similarity Grouper (Deep Learning)")

uploaded_files = st.file_uploader(
    "Upload multiple images",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    # Create working folder
    input_folder = "uploaded_images"
    os.makedirs(input_folder, exist_ok=True)

    # Save uploaded files
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(input_folder, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)

    # ==== Load or Build Cache ====
    if os.path.exists(cache_file):
        cache = np.load(cache_file, allow_pickle=True).item()
    else:
        cache = {}

    features_list, valid_paths = [], []
    for path in file_paths:
        if path in cache:
            features_list.append(cache[path])
            valid_paths.append(path)
        else:
            feat = extract_features(path)
            if feat is not None:
                cache[path] = feat
                features_list.append(feat)
                valid_paths.append(path)

    np.save(cache_file, cache)
    features_array = np.array(features_list)

    # ==== Clustering ====
    st.write("🔍 Clustering images...")
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='euclidean')
    labels = dbscan.fit_predict(features_array)

    # ==== Display Groups in Streamlit ====
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        group_name = f"Group {label+1}" if label != -1 else "Ungrouped"
        st.subheader(group_name)
        cols = st.columns(5)
        idx = 0
        for img_path, lbl in zip(valid_paths, labels):
            if lbl == label:
                with cols[idx % 5]:
                    st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)
                idx += 1
