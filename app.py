import os
import shutil
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.cluster import DBSCAN
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# ==== CONFIG ====
resize_dim = (224, 224)
eps_value = 0.5
min_samples_value = 2
cache_file = "image_features.npy"

# ==== Load Model ====
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()  # remove classifier
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ==== Preprocessing ====
transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_t)
        return features.cpu().numpy().flatten()
    except:
        return None

# ==== UI ====
st.title("🖼 Image Similarity Grouper (PyTorch Version)")

uploaded_files = st.file_uploader(
    "Upload multiple images",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    input_folder = "uploaded_images"
    os.makedirs(input_folder, exist_ok=True)
    file_paths = []
    for file in uploaded_files:
        path = os.path.join(input_folder, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(path)

    # ==== Cache ====
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

    # ==== Display Groups ====
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
