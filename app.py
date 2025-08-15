import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
from io import BytesIO
from sklearn.cluster import DBSCAN
import cv2

# Load ResNet50 model for general image feature extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

st.title("Image Similarity Clustering App")

# Initialize session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# File uploader
uploaded = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Add new uploads to session state
if uploaded:
    st.session_state.uploaded_files.extend(uploaded)

# Clear All button
if st.button("🗑️ Clear All"):
    st.session_state.uploaded_files = []
    st.stop()

def get_color_histogram(img_path):
    """
    Extracts a color histogram from an image.
    This provides a feature vector that is highly sensitive to color differences.
    We use the HSV color space which is generally more robust to lighting changes.
    """
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(256 * 3) # Return a zero vector if image fails to load
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for H, S, and V channels
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Normalize and flatten histograms
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

def extract_features(img_path):
    """
    Extracts a combined feature vector: visual features from ResNet50 and color features.
    """
    # ResNet50 features
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    resnet_features = model.predict(x).flatten()
    
    # Color histogram features
    color_features = get_color_histogram(img_path)
    
    # Concatenate the features to create a single, richer feature vector
    return np.concatenate([resnet_features, color_features])

def get_common_cluster_name(filenames):
    """
    Generates a common name for a cluster based on the filenames.
    """
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
    # Save uploaded files temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in st.session_state.uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path)

    # Extract features for all images
    # We now get the combined visual and color features
    features = [extract_features(path) for path in file_paths]
    features = np.array(features)

    # Compute similarity
    sim_matrix = cosine_similarity(features)

    # Clamp values to ensure they are within a valid range
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

    # Convert similarity to a distance matrix
    dist_matrix = 1 - sim_matrix
    
    # Clustering using DBSCAN
    # A small 'eps' value is used to force tighter clusters.
    dbscan = DBSCAN(eps=0.05, min_samples=1, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)

    # Create clusters from DBSCAN labels
    clusters_dict = {}
    for i, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(i)
    
    clusters = list(clusters_dict.values())
    
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
    if not clusters:
        st.info("No clusters found. Try adjusting the 'eps' value or uploading more images.")
    else:
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
