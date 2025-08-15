import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
# The following two imports for ResNet50 have been removed to solve the ModuleNotFoundError.
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
import pandas as pd
from io import BytesIO
from sklearn.cluster import DBSCAN

# The ResNet50 model and feature extraction is removed to resolve the TensorFlow dependency.
# The script will now rely solely on color features for clustering.
# model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

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
    Extracts a simple RGB color histogram from an image using PIL.
    This replaces the cv2-based function and ensures the script can run without it.
    The histogram provides a feature vector sensitive to color differences.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        
        # Resize image to a small size to speed up histogram calculation
        img_resized = img.resize((32, 32))
        
        # Get the histogram from the image data
        hist = img_resized.histogram()
        
        # Normalize the histogram
        hist = np.array(hist, dtype=np.float32)
        hist /= hist.sum()
        
        return hist
    except Exception as e:
        st.error(f"Error processing image {os.path.basename(img_path)}: {e}")
        return np.zeros(256 * 3)

def extract_features(img_path):
    """
    Extracts only the color histogram as the feature vector.
    The ResNet50-based feature extraction has been removed.
    """
    # Color histogram features
    color_features = get_color_histogram(img_path)
    
    # Return the color features directly as the full feature vector
    return color_features

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
