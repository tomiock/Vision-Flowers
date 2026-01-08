import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import random

# --- Configuration (Corrected Paths) ---
BASE_PATH = '/data-net/storage2/datasets/OxfordF'

# Path to images folder (updated to include 'jpg' subfolder)
IMAGES_DIR = os.path.join(BASE_PATH, 'images/jpg')

# Path to labels file (updated to 'labels.npz')
LABELS_FILE = os.path.join(BASE_PATH, 'labels.npz')

# Path to category mapping (remains the same)
CAT_MAP_FILE = os.path.join(BASE_PATH, 'cat_to_name.json')

# --- 1. Load Data (Cached for speed) ---
@st.cache_data
def load_dataset():
    """
    Loads image paths, ground truth labels, and category names.
    """
    # 1. Load Category Mapping (e.g., "1" -> "pink primrose")
    # Verify file exists first to avoid crash if path is slightly off
    if not os.path.exists(CAT_MAP_FILE):
        st.error(f"Category map not found at: {CAT_MAP_FILE}")
        return [], {}

    with open(CAT_MAP_FILE, 'r') as f:
        cat_to_name = json.load(f)
    
    # 2. Load Ground Truth Labels
    if not os.path.exists(LABELS_FILE):
        st.error(f"Labels file not found at: {LABELS_FILE}")
        return [], {}
        
    # Using 'arr_0' as per your dataset class
    labels_raw = np.load(LABELS_FILE)['arr_0']
    
    # 3. Get Image Files 
    # CRITICAL: Must be sorted to align with the labels array indices
    if not os.path.exists(IMAGES_DIR):
        st.error(f"Images directory not found at: {IMAGES_DIR}")
        return [], {}
        
    image_files = sorted(os.listdir(IMAGES_DIR))
    
    # 4. Create a structured list of dictionaries
    dataset = []
    
    # Flatten labels just in case shape is (N, 1)
    labels_flat = labels_raw.reshape(-1)

    for idx, filename in enumerate(image_files):
        # Safety check to prevent index errors if files > labels
        if idx >= len(labels_flat):
            break
            
        label_idx = str(labels_flat[idx])
        class_name = cat_to_name.get(label_idx, "Unknown")
        
        dataset.append({
            "filename": filename,
            "path": os.path.join(IMAGES_DIR, filename),
            "label_idx": label_idx,
            "class_name": class_name
        })
        
    return dataset, cat_to_name

# --- 2. Main App Logic ---
def main():
    st.set_page_config(page_title="Flower Dataset Viewer", layout="wide")
    st.title("🌸 Oxford Flowers Dataset Explorer")

    dataset, cat_map = load_dataset()
    
    if not dataset:
        st.warning("No data loaded. Please check the paths in the script.")
        st.stop()

    # --- Sidebar: Controls ---
    st.sidebar.header("Filter Options")
    
    # Filter by Class Name
    all_classes = sorted(list(set(d['class_name'] for d in dataset)))
    selected_class = st.sidebar.selectbox("Select Flower Type", ["All"] + all_classes)
    
    # Filter Logic
    if selected_class != "All":
        filtered_data = [d for d in dataset if d['class_name'] == selected_class]
    else:
        filtered_data = dataset

    st.sidebar.markdown(f"**Total Images:** {len(dataset)}")
    st.sidebar.markdown(f"**Showing:** {len(filtered_data)}")

    # Visualization Mode
    view_mode = st.radio("View Mode", ["Grid View", "Single Focus"], horizontal=True)

    # --- 3. Visualization ---
    
    if view_mode == "Grid View":
        # Pagination for Grid
        page_size = 20
        total_pages = max(1, len(filtered_data) // page_size + (1 if len(filtered_data) % page_size > 0 else 0))
        page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        batch = filtered_data[start_idx:end_idx]

        # Display Grid (4 columns)
        cols = st.columns(4)
        for i, item in enumerate(batch):
            with cols[i % 4]:
                try:
                    img = Image.open(item['path'])
                    st.image(img, use_container_width=True)
                    st.caption(f"**{item['class_name']}**\n(ID: {item['label_idx']})")
                except Exception as e:
                    st.error(f"Err loading {item['filename']}")

    elif view_mode == "Single Focus":
        if st.button("Shuffle & Pick Random"):
            item = random.choice(filtered_data)
            st.session_state['focused_item'] = item
        
        if 'focused_item' not in st.session_state:
            st.session_state['focused_item'] = filtered_data[0] if filtered_data else None
            
        item = st.session_state['focused_item']
        
        if item:
            c1, c2 = st.columns([1, 1])
            with c1:
                img = Image.open(item['path'])
                st.image(img, use_container_width=True)
            with c2:
                st.header(item['class_name'])
                st.markdown(f"**File Name:** `{item['filename']}`")
                st.markdown(f"**Label ID:** `{item['label_idx']}`")

if __name__ == "__main__":
    main()
