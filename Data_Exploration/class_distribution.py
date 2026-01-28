import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from scipy.stats import gaussian_kde

# -----------------------------
# Paths
# -----------------------------
ROOT = "./OxfordF"
IMAGES_DIR = os.path.join(ROOT, "images/jpg")
LABELS_FILE = os.path.join(ROOT, "labels.npz")
CAT_JSON = os.path.join(ROOT, "cat_to_name.json")

# -----------------------------
# Parameters
# -----------------------------
TARGET_CLASS = 48   # <-- Change class ID here (1–102)
MAX_PIXELS_PER_IMAGE = 5000  # Limit pixels per image for KDE
RESIZE_DIM = (128, 128)      # Resize large images to this size

# -----------------------------
# Load data
# -----------------------------
labels = np.load(LABELS_FILE)['arr_0'].reshape(-1)

with open(CAT_JSON, "r") as f:
    cat2name = json.load(f)

image_filenames = sorted(os.listdir(IMAGES_DIR))

# -----------------------------
# Collect HSV values efficiently
# -----------------------------
hues, sats, vals = [], [], []

for idx, filename in enumerate(image_filenames):
    if labels[idx] != TARGET_CLASS:
        continue

    img_path = os.path.join(IMAGES_DIR, filename)
    image = Image.open(img_path).convert("RGB")
    
    image = image.resize(RESIZE_DIM)
    img_np = np.array(image) / 255.0
    hsv = rgb_to_hsv(img_np)

    # Sample pixels per channel to reduce computation
    for channel, arr in zip([hues, sats, vals], hsv.transpose(2, 0, 1)):
        pixels = arr.ravel()
        if len(pixels) > MAX_PIXELS_PER_IMAGE:
            pixels = np.random.choice(pixels, MAX_PIXELS_PER_IMAGE, replace=False)
        channel.extend(pixels)

hues = np.array(hues)
sats = np.array(sats)
vals = np.array(vals)

# -----------------------------
# KDE computation
# -----------------------------
x = np.linspace(0.0, 1.0, 500)
kde_h = gaussian_kde(hues)
kde_s = gaussian_kde(sats)
kde_v = gaussian_kde(vals)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(x, kde_h(x), label="Hue", linewidth=2)
plt.plot(x, kde_s(x), label="Saturation", linewidth=2)
plt.plot(x, kde_v(x), label="Value (Illuminance)", linewidth=2)

plt.fill_between(x, 0, kde_h(x), color="blue", alpha=0.2)
plt.fill_between(x, 0, kde_s(x), color="orange", alpha=0.2)
plt.fill_between(x, 0, kde_v(x), color="green", alpha=0.2)

plt.xlim(0.0, 1.0)        
plt.ylim(bottom=0.0)     
plt.title(f"HSV KDE Distribution - {cat2name[str(TARGET_CLASS)]}", fontsize=13)
plt.xlabel("HSV Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

save_path = os.path.join(ROOT, f"hsv_kde_class_{TARGET_CLASS}.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved plot to {save_path}")
