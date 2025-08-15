import streamlit as st
import os
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from itertools import combinations

# --- Load ResNet50 (no classifier head) ---
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove last fc layer
    model.eval()
    return model

# --- Image transforms for ResNet ---
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --- Extract ResNet features ---
def extract_deep_features(img, model):
    with torch.no_grad():
        features = model(preprocess_image(img))
    return features.squeeze().numpy().flatten()

# --- Extract color histogram (HSV) ---
def extract_color_histogram(img):
    img_cv = np.array(img)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --- Calculate combined similarity ---
def combined_similarity(feat1, feat2, hist1, hist2):
    shape_sim = cosine_similarity([feat1], [feat2])[0][0]
    color_sim = cv2.compareHist(hist1.astype("float32"), hist2.astype("float32"), cv2.HISTCMP_CORREL)
    return (shape_sim + color_sim) / 2  # equal weight to shape & color

# --- Group images based on similarity ---
def group_images(images, threshold=0.9):
    model = load_model()
    features = []
    colors = []
    for img in images:
        features.append(extract_deep_features(img, model))
        colors.append(extract_color_histogram(img))

    groups = []
    visited = set()

    for i in range(len(images)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, len(images)):
            if j in visited:
                continue
            sim = combined_similarity(features[i], features[j], colors[i], colors[j])
            if sim >= threshold:
                group.append(j)
                visited.add(j)
        groups.append(group)
    return groups

# --- Streamlit UI ---
st.title("📸 Image Similarity Clustering (90% Match with Color Sensitivity)")
uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    images = [Image.open(file).convert("RGB") for file in uploaded_files]
    groups = group_images(images, threshold=0.9)

    st.subheader("Clustering Results")
    for idx, group in enumerate(groups, 1):
        st.markdown(f"**Group {idx}:**")
        cols = st.columns(len(group))
        for col, img_idx in zip(cols, group):
            col.image(images[img_idx], use_column_width=True)
