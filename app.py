"""
Streamlit app: Upload images and group them by visual similarity.

How it works
------------
1) You upload multiple images (PNG/JPG/WebP).
2) The app extracts a compact feature vector for each image using a pretrained
   computer-vision model (ResNet50) from torchvision.
3) It clusters the feature vectors using Agglomerative Clustering with a
   cosine distance threshold (you control it via a slider).
4) It shows each cluster with the images that belong to it.

Run locally
-----------
1) Save this file as `app.py` (or keep the current name and adjust the command).
2) Create & activate a virtual env (optional but recommended).
3) Install deps:
   pip install streamlit torch torchvision pillow scikit-learn numpy
4) Start the app:
   streamlit run app.py

Notes
-----
- Works on CPU; GPU is optional.
- If you have many large images, set the "Max image side" slider to a smaller
  value to speed up processing.
- The distance threshold controls how tight clusters are. Lower = more
  clusters; higher = fewer clusters.
"""

import io
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# ----------------------------
# UI — Sidebar controls
# ----------------------------
st.set_page_config(page_title="Image Grouper: Visual Similarity", layout="wide")
st.title("👀 Image Grouper — Group by Visual Similarity")
st.caption("Upload images and automatically group similar-looking ones.")

with st.sidebar:
    st.header("Settings")
    max_side = st.slider("Max image side (px) for feature extraction", 128, 1024, 384, 32)
    pool_type = st.selectbox("Feature pooling", ["avg", "gem"], index=0, help="How we pool the CNN feature map into a vector.")
    dist_thresh = st.slider(
        "Cosine distance threshold (clustering)",
        min_value=0.05,
        max_value=0.90,
        value=0.30,
        step=0.01,
        help=(
            "Lower = stricter clustering (more clusters).\n"
            "Higher = looser clustering (fewer clusters).\n"
            "We use agglomerative clustering with this distance threshold."
        ),
    )
    min_cluster_size = st.number_input("Minimum cluster size (hide tiny groups)", min_value=1, value=1, step=1)
    show_embeddings = st.checkbox("Show feature vectors (debug)", value=False)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def _load_image_bytes(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

@st.cache_resource(show_spinner=False)
def _load_backbone():
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    # remove classification head => global feature map from layer4
    body = nn.Sequential(*list(model.children())[:-2])  # B x 2048 x H x W
    body.eval()
    for p in body.parameters():
        p.requires_grad_(False)
    preprocess = weights.transforms()
    return body, preprocess

class GeMPool(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.mean(x, dim=(-1, -2)).pow(1.0 / self.p)
        return x

@st.cache_resource(show_spinner=False)
def _feature_extractor(pool: str = "avg"):
    body, preprocess = _load_backbone()
    if pool == "gem":
        pool_layer = GeMPool()
    else:
        pool_layer = nn.AdaptiveAvgPool2d((1, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body.to(device)
    pool_layer.to(device)
    return body, pool_layer, preprocess, device

@torch.inference_mode()
def _extract_features(images: List[Image.Image], max_side: int, pool: str) -> np.ndarray:
    body, pool_layer, preprocess, device = _feature_extractor(pool)

    # custom resize keeping aspect ratio so longest side == max_side
    def ratio_resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = max_side / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return img.resize((new_w, new_h), Image.BILINEAR)

    feats = []
    for img in images:
        small = ratio_resize(img)
        # torchvision weights.transforms() expects 224 crop; we center pad to square
        w, h = small.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(small, ((side - w) // 2, (side - h) // 2))
        x = preprocess(canvas).unsqueeze(0).to(device)
        fmap = body(x)  # B x 2048 x H x W
        if isinstance(pool_layer, nn.AdaptiveAvgPool2d):
            v = pool_layer(fmap).flatten(1)
        else:
            v = pool_layer(fmap).flatten(1)
        v = torch.nn.functional.normalize(v, dim=1)
        feats.append(v.cpu().numpy()[0])
    return np.stack(feats, axis=0)

# ----------------------------
# File upload
# ----------------------------
files = st.file_uploader(
    "Upload images (PNG, JPG, JPEG, WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if not files:
    st.info("⬆️ Upload 3–50 images to get started.")
    st.stop()

if len(files) < 2:
    st.warning("Upload at least 2 images to form clusters.")

# Load images
pil_images: List[Image.Image] = []
filenames: List[str] = []
for f in files:
    try:
        img = _load_image_bytes(f.read())
        pil_images.append(img)
        filenames.append(f.name)
    except Exception as e:
        st.error(f"Failed to read {f.name}: {e}")

# Extract features
with st.spinner("Extracting visual features…"):
    feats = _extract_features(pil_images, max_side=max_side, pool=pool_type)

# Compute cosine distance matrix (0 = identical, 2 = opposite for normalized vectors)
D = cosine_distances(feats)

# Agglomerative clustering with distance threshold
with st.spinner("Clustering images…"):
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_thresh,
    )
    labels = clustering.fit_predict(feats)

# Group by labels
clusters = {}
for idx, lab in enumerate(labels):
    clusters.setdefault(int(lab), []).append(idx)

# Optionally filter small clusters
if min_cluster_size > 1:
    clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}

st.success(f"Found {len(clusters)} clusters")

# Display clusters
for cid, members in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    st.subheader(f"Cluster {cid} · {len(members)} image(s)")
    cols = st.columns(6)
    i = 0
    for idx in members:
        with cols[i % 6]:
            st.image(pil_images[idx], caption=filenames[idx], use_column_width=True)
        i += 1

# Optional: show embeddings table (rounded)
if show_embeddings:
    import pandas as pd
    df = pd.DataFrame(feats, index=filenames)
    st.dataframe(df.round(4))

# Download labels as CSV
import pandas as pd
label_df = pd.DataFrame({"filename": filenames, "cluster": labels})
st.download_button(
    "Download cluster assignments (CSV)",
    data=label_df.to_csv(index=False).encode("utf-8"),
    file_name="image_clusters.csv",
    mime="text/csv",
)
