import io
import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

st.set_page_config(page_title="Image Grouper", layout="wide")
st.title("🖼️ Image Grouper — Lightweight ResNet18 Version")
st.caption("Upload images and group them by visual similarity.")

# Sidebar controls
with st.sidebar:
    max_side = st.slider("Max image side (px)", 128, 512, 256, 32)
    dist_thresh = st.slider("Cosine distance threshold", 0.05, 0.9, 0.3, 0.01)
    min_cluster_size = st.number_input("Minimum cluster size", 1, 10, 1)

# Cache model
@st.cache_resource
def load_model():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    backbone = nn.Sequential(*list(model.children())[:-2])  # remove head
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    preprocess = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    return backbone, preprocess, device

# Cache image load
@st.cache_data
def load_image(file_bytes):
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

# Extract features
@torch.inference_mode()
def extract_features(images, max_side):
    backbone, preprocess, device = load_model()

    def resize_keep_ratio(img):
        w, h = img.size
        scale = max_side / max(w, h)
        return img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    feats = []
    for img in images:
        img_small = resize_keep_ratio(img)
        w, h = img_small.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(img_small, ((side - w)//2, (side - h)//2))
        x = preprocess(canvas).unsqueeze(0).to(device)
        fmap = backbone(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
        v = torch.nn.functional.normalize(pooled, dim=1)
        feats.append(v.cpu().numpy()[0])
    return np.stack(feats, axis=0)

# File uploader
files = st.file_uploader(
    "Upload images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)

if not files:
    st.info("⬆️ Upload at least 2 images")
    st.stop()

if len(files) < 2:
    st.warning("Need at least 2 images to cluster.")

# Load images
pil_images = []
filenames = []
for f in files:
    try:
        img = load_image(f.read())
        pil_images.append(img)
        filenames.append(f.name)
    except Exception as e:
        st.error(f"Error loading {f.name}: {e}")

# Extract features
with st.spinner("Extracting features…"):
    feats = extract_features(pil_images, max_side=max_side)

# Cluster
D = cosine_distances(feats)
with st.spinner("Clustering…"):
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=dist_thresh
    )
    labels = clustering.fit_predict(feats)

# Group results
clusters = {}
for idx, lab in enumerate(labels):
    clusters.setdefault(lab, []).append(idx)

if min_cluster_size > 1:
    clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

st.success(f"Found {len(clusters)} clusters.")

# Show clusters
for cid, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
    st.subheader(f"Cluster {cid} — {len(members)} images")
    cols = st.columns(6)
    for i, idx in enumerate(members):
        with cols[i % 6]:
            st.image(pil_images[idx], caption=filenames[idx], use_column_width=True)

# Download CSV
import pandas as pd
label_df = pd.DataFrame({"filename": filenames, "cluster": labels})
st.download_button(
    "Download CSV",
    data=label_df.to_csv(index=False).encode("utf-8"),
    file_name="clusters.csv",
    mime="text/csv",
)
