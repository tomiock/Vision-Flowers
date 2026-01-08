import argparse
import os
import io
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from dataset import FlowersDataset

# --- Configuration ---
TEACHER_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
NUM_SHARDS = 4  # You requested 4 files
BATCH_SIZE = 256

def get_teacher_reasoning_batch(model, processor, images, true_labels, device):
    """
    Generates descriptions for a batch of images.
    """
    prompts = []
    for label in true_labels:
        # Ground-Truth Guided Distillation Prompt
        text_prompt = (
            f"You are a botanical expert. This image shows a **{label}** flower. "
            f"Describe the specific visual features visible in this image (such as petal shape, color, texture, stamen, or leaves) "
            f"that confirm it is a {label}. Be concise and factual."
        )
        prompts.append(text_prompt)

    messages_batch = []
    for img, txt in zip(images, prompts):
        messages_batch.append([
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}
        ])

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]

    inputs = processor(
        text=texts, images=images, padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=False 
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return output_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF")
    parser.add_argument("--output_dir", type=str, default="/data-net/storage2/datasets/OxfordF/parquet_shards")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Teacher: {TEACHER_MODEL_ID}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID)

    # Load Dataset (Raw Images)
    dataset = FlowersDataset(
        images_dir=os.path.join(args.data_root, 'images/jpg'),
        labels_file=os.path.join(args.data_root, 'labels.npz'),
        cat_json=os.path.join(args.data_root, 'cat_to_name.json'),
        preprocess=lambda x: x,
        prompt_template="{}"
    )

    total_samples = len(dataset)
    samples_per_shard = int(np.ceil(total_samples / NUM_SHARDS))
    
    print(f"Generating {total_samples} samples across {NUM_SHARDS} parquet shards...")

    current_shard_data = []
    current_shard_idx = 0
    
    # Batch accumulation lists
    batch_images = []
    batch_labels = []
    
    # We iterate through the dataset and flush to parquet every N samples
    for i in tqdm(range(total_samples)):
        img, _, label_idx = dataset[i]
        true_label = dataset.classes[label_idx]
        
        batch_images.append(img)
        batch_labels.append(true_label)
        
        # Process batch if full or at end of dataset
        if len(batch_images) == BATCH_SIZE or i == total_samples - 1:
            descriptions = get_teacher_reasoning_batch(model, processor, batch_images, batch_labels, device)
            
            # Format entries for Parquet
            for img_obj, label, desc in zip(batch_images, batch_labels, descriptions):
                
                # 1. Convert Image to Bytes (required by your class)
                img_byte_arr = io.BytesIO()
                img_obj.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                # Structure matches: row['images'] -> list of dicts with 'bytes'
                images_field = [{'bytes': img_bytes}]
                
                # 2. Format Texts (required by your class)
                # Structure matches: row['texts'] -> list of dicts with 'user', 'assistant'
                
                # "Answer First" Format for Fast Eval
                assistant_response = f"**{label}**.\n{desc}"
                
                texts_field = [
                    {
                        "user": "Identify the specific type of this flower. Answer concisely.",
                        "assistant": assistant_response
                    }
                ]
                
                current_shard_data.append({
                    "images": images_field,
                    "texts": texts_field
                })
            
            # Reset Batch
            batch_images = []
            batch_labels = []

        # Check if we need to save a shard
        if len(current_shard_data) >= samples_per_shard or i == total_samples - 1:
            if len(current_shard_data) > 0:
                shard_filename = f"flowers_{current_shard_idx:03d}.parquet"
                save_path = os.path.join(args.output_dir, shard_filename)
                
                print(f"Saving Shard {current_shard_idx} to {save_path} ({len(current_shard_data)} samples)...")
                
                df = pd.DataFrame(current_shard_data)
                df.to_parquet(save_path)
                
                current_shard_data = [] # Reset for next shard
                current_shard_idx += 1

    print("Done. All parquet shards generated.")

if __name__ == "__main__":
    main()