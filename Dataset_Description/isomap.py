import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import Isomap
from matplotlib.colors import rgb_to_hsv
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -----------------------------
# Paths
# -----------------------------
ROOT = "./OxfordF"
IMAGES_DIR = os.path.join(ROOT, "images/jpg")
LABELS_FILE = os.path.join(ROOT, "labels.npz")
CAT_JSON = os.path.join(ROOT, "cat_to_name.json")

# -----------------------------
# Load data
# -----------------------------
labels = np.load(LABELS_FILE)['arr_0'].reshape(-1)

with open(CAT_JSON, "r") as f:
    cat2name = json.load(f)

image_filenames = sorted(os.listdir(IMAGES_DIR))

# -----------------------------
# Center crop
# -----------------------------
def center_crop(image, size=96):
    w, h = image.size
    crop_size = min(w, h)

    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))
    image = image.resize((size, size), Image.BILINEAR)
    return image

# -----------------------------
# Extract HSV features + store one image per class
# -----------------------------
class_features = {}
class_images = {}

for idx, filename in enumerate(image_filenames):
    img_path = os.path.join(IMAGES_DIR, filename)
    image = Image.open(img_path).convert("RGB")
    image = center_crop(image, size=96)

    img_np = np.array(image) / 255.0
    hsv = rgb_to_hsv(img_np)
    mean_hsv = hsv.mean(axis=(0, 1))

    class_id = labels[idx]
    class_features.setdefault(class_id, []).append(mean_hsv)

    # Store ONE representative image per class (first occurrence)
    if class_id not in class_images:
        class_images[class_id] = image

# -----------------------------
# Compute class centroids
# -----------------------------
centroids = []
rep_images = []

for class_id in sorted(class_features.keys()):
    centroid = np.mean(class_features[class_id], axis=0)
    centroids.append(centroid)
    rep_images.append(class_images[class_id])

centroids = np.array(centroids)

# -----------------------------
# ISOMAP on centroids
# -----------------------------
isomap = Isomap(n_neighbors=8, n_components=2)
embedding = isomap.fit_transform(centroids)

# -----------------------------
# Plot images at centroid locations
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 14))

ax.set_title("ISOMAP of Oxford Flowers 102 (Class Centroids, HSV)", fontsize=14)
ax.set_xlabel("ISOMAP Dimension 1")
ax.set_ylabel("ISOMAP Dimension 2")

x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

padding_x = 0.1 * (x_max - x_min)
padding_y = 0.1 * (y_max - y_min)

ax.set_xlim(x_min - padding_x, x_max + padding_x)
ax.set_ylim(y_min - padding_y, y_max + padding_y)

for (x, y), img in zip(embedding, rep_images):
    imagebox = OffsetImage(np.array(img), zoom=0.4)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

plt.tight_layout()

save_path = os.path.join(ROOT, "isomap_hsv_class_images.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved to {save_path}")