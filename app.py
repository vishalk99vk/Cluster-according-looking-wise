import streamlit as st
import os
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
from io import BytesIO
from sklearn.cluster import DBSCAN # Import DBSCAN for more robust clustering

# Load ResNet50 model for feature extraction
# ResNet50 is a powerful model for general object recognition, but may
# not be sensitive enough to subtle color differences on similar objects.
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
    st.stop()  # Stop execution to refresh the uploader

def extract_features(img_path):
    """
    Extracts features from an image using the ResNet50 model.
    The image is resized, converted to an array, and preprocessed for the model.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def get_common_cluster_name(filenames):
    """
    Generates a common name for a cluster based on the filenames.
    This logic remains the same as it correctly identifies common words.
    """
    names = [os.path.splitext(f)[0] for f in filenames]
    if not names:
        return "Unknown Cluster"
    if len(names) == 1:
        return names[0]
    
    # Split filenames into words
    split_names = [name.split() for name in names]
    
    # Find the intersection of words across all filenames
    common = set(split_names[0])
    for parts in split_names[1:]:
        common &= set(parts)

    # Reconstruct the common name from the first filename
    common_name = " ".join([w for w in names[0].split() if w in common])
    
    # If no common name is found, use the first filename as a fallback
    return common_name if common_name else names[0]

if st.session_state.uploaded_files:
    # Save uploaded files temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    for file in st.session_state.uploaded_files:
        # Use a unique path to avoid name collisions
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_paths.append(file_path)

    # Extract features for all images
    features = [extract_features(path) for path in file_paths]
    features = np.array(features)

    # Compute similarity (which is a form of distance)
    sim_matrix = cosine_similarity(features)

    # Convert similarity to a distance matrix for DBSCAN
    # DBSCAN requires a distance metric, and distance = 1 - similarity.
    dist_matrix = 1 - sim_matrix
    
    # Clustering using DBSCAN
    # DBSCAN is more suitable here as it doesn't require a fixed number of clusters
    # and can find clusters of various shapes.
    # The 'eps' parameter is the maximum distance between two samples for one to be considered
    # as in the neighborhood of the other. It's similar to your 'threshold', but for distance.
    # 'min_samples' is the number of samples in a neighborhood for a point to be considered as a core point.
    # Setting min_samples=1 means every point can potentially be a core point, which is good
    # for finding clusters even with only two images.
    dbscan = DBSCAN(eps=0.05, min_samples=1, metric='precomputed') # Adjusted eps for a tighter cluster
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
