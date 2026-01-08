import os
import io
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

def main():
    # --- Configuration ---
    ROOT = "/data-net/storage2/datasets/OxfordF"
    IMAGES_DIR = os.path.join(ROOT, "images/jpg")
    LABELS_FILE = os.path.join(ROOT, "labels.npz")
    CAT_JSON = os.path.join(ROOT, "cat_to_name.json")
    SPLITS_PATH = os.path.join(ROOT, "my_data.json")
    
    # Where to save the parquet files
    OUTPUT_DIR = os.path.join(ROOT, "parquet_simple_brief_train")
    NUM_SHARDS = 4  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading metadata...")
    
    # 1. Load Labels, Mappings & Splits
    labels = np.load(LABELS_FILE)['arr_0'].reshape(-1)
    
    with open(CAT_JSON, 'r') as f:
        cat2name = json.load(f)
        
    with open(SPLITS_PATH, 'rb') as f:
        splits = json.load(f)

    # 2. Define Training Indices
    # Based on your previous scripts: Training = 'valid' + 'tstid' lists
    train_indices_1based = splits['valid'] + splits['tstid']
    
    # Convert to 0-based indexing if necessary (assuming split file uses 1-based)
    if min(train_indices_1based) == 1:
        train_indices = {x - 1 for x in train_indices_1based}
    else:
        train_indices = set(train_indices_1based)

    print(f"Total training samples selected: {len(train_indices)}")

    image_filenames = sorted(os.listdir(IMAGES_DIR))
    
    if len(image_filenames) != len(labels):
        print(f"Warning: {len(image_filenames)} images vs {len(labels)} labels.")

    # Calculate shard size based on filtered count
    total_samples = len(train_indices)
    samples_per_shard = int(np.ceil(total_samples / NUM_SHARDS))
    
    print(f"Processing {total_samples} samples into {NUM_SHARDS} shards (CPU only)...")

    current_shard_data = []
    current_shard_idx = 0
    samples_processed = 0
    
    # 3. Iterate and Filter
    for idx, filename in enumerate(tqdm(image_filenames)):
        # Skip if this index is not in our training set
        if idx not in train_indices:
            continue
            
        label_id = str(labels[idx])
        class_name = cat2name[label_id]
        image_path = os.path.join(IMAGES_DIR, filename)
        
        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            
            images_field = [{'bytes': img_bytes}]
            
            texts_field = [
                {
                    "user": "What type of flower is this? Answer briefly",
                    "assistant": class_name
                }
            ]
            
            current_shard_data.append({
                "images": images_field,
                "texts": texts_field
            })
            
            samples_processed += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        # 4. Save Shard
        if len(current_shard_data) >= samples_per_shard or samples_processed == total_samples:
            if len(current_shard_data) > 0:
                shard_filename = f"flowers_train_{current_shard_idx:03d}.parquet"
                save_path = os.path.join(OUTPUT_DIR, shard_filename)
                
                print(f"Saving {shard_filename} ({len(current_shard_data)} samples)...")
                
                df = pd.DataFrame(current_shard_data)
                df.to_parquet(save_path)
                
                current_shard_data = []
                current_shard_idx += 1

    print(f"Done. Training dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()