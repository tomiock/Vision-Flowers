import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------
# Paths
# -----------------------------
ROOT = "./OxfordF"
LABELS_FILE = os.path.join(ROOT, "labels.npz")
CAT_JSON = os.path.join(ROOT, "cat_to_name.json")
SPLITS_PATH = os.path.join(ROOT, "my_data.json")

# -----------------------------
# Load data
# -----------------------------
labels = np.load(LABELS_FILE)['arr_0'].reshape(-1)

with open(CAT_JSON, "r") as f:
    cat2name = json.load(f)

label_counts = Counter(labels)

sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
sorted_labels, counts = zip(*sorted_items)
names = [cat2name[str(label)] for label in sorted_labels]

# -----------------------------
# Plot number of samples 
# -----------------------------

fig, ax = plt.subplots(figsize=(18, 7), layout="constrained")
bars = ax.bar(range(len(names)), counts, color='skyblue', width=0.8)
ax.set_xlim(-0.5, len(names) - 0.5)
ax.set_ylim(0, max(counts) * 1.1) 

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=90, fontsize=8)
ax.set_ylabel("Number of images", fontsize=12)
ax.set_title("Class Distribution in Oxford Flowers 102 Dataset", fontsize=14, pad=20)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

save_path = os.path.join(ROOT, "class_distribution.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')

plt.show()

print(f"Plot saved to {save_path}")



# --------------------------------
# Print number of images per split
# --------------------------------
with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)

# Each split
train_split = splits.get("trnid", [])
valid_split = splits.get("valid", [])
test_split  = splits.get("tstid", [])

print(f"Number of training images (trnid): {len(train_split)}")
print(f"Number of validation images (valid): {len(valid_split)}")
print(f"Number of test images (tstid): {len(test_split)}")

# Optional: total images in splits
total_split_images = len(set(train_split + valid_split + test_split))
print(f"Total number of images accounted in splits: {total_split_images}")
