import argparse
import os
import io
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from dataset import FlowersDataset

TEACHER_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
NUM_SHARDS = 4
BATCH_SIZE = 256

def get_teacher_reasoning_batch(model, processor, images, true_labels, device):
    """
    Generates descriptions for a batch of images.
    """
    prompts = []
    for label in true_labels:
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
    parser.add_argument("--output_dir", type=str, default="/data-net/storage2/datasets/OxfordF/parquet_shards_complex_train")
    parser.add_argument("--splits_path", type=str, default="/data-net/storage2/datasets/OxfordF/my_data.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Teacher: {TEACHER_MODEL_ID}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        TEACHER_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(TEACHER_MODEL_ID)

    dataset = FlowersDataset(
        images_dir=os.path.join(args.data_root, 'images/jpg'),
        labels_file=os.path.join(args.data_root, 'labels.npz'),
        cat_json=os.path.join(args.data_root, 'cat_to_name.json'),
        preprocess=lambda x: x,
        prompt_template="{}"
    )

    print(f"Loading splits from {args.splits_path}...")
    with open(args.splits_path, 'rb') as f:
        splits = json.load(f)

    train_indices_1based = splits['valid'] + splits['tstid']
    
    if min(train_indices_1based) == 1:
        train_indices = {x - 1 for x in train_indices_1based}
    else:
        train_indices = set(train_indices_1based)

    total_training_samples = len(train_indices)
    samples_per_shard = int(np.ceil(total_training_samples / NUM_SHARDS))
    
    print(f"Generating {total_training_samples} training samples across {NUM_SHARDS} parquet shards...")

    current_shard_data = []
    current_shard_idx = 0
    
    batch_images = []
    batch_labels = []
    
    samples_processed = 0

    for i in tqdm(range(len(dataset))):
        
        if i not in train_indices:
            continue
            
        img, _, label_idx = dataset[i]
        true_label = dataset.classes[label_idx]
        
        batch_images.append(img)
        batch_labels.append(true_label)
        
        is_last_batch = (samples_processed + len(batch_images) == total_training_samples)
        
        if len(batch_images) == BATCH_SIZE or is_last_batch:
            descriptions = get_teacher_reasoning_batch(model, processor, batch_images, batch_labels, device)
            
            for img_obj, label, desc in zip(batch_images, batch_labels, descriptions):
                
                img_byte_arr = io.BytesIO()
                img_obj.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                images_field = [{'bytes': img_bytes}]
                
                assistant_response = f"{desc} It is a {label}"
                
                texts_field = [
                    {
                        "user": "Describe the flower present on the image and specify its type.",
                        "assistant": assistant_response
                    }
                ]
                
                current_shard_data.append({
                    "images": images_field,
                    "texts": texts_field
                })
            
            samples_processed += len(batch_images)
            
            batch_images = []
            batch_labels = []

        if len(current_shard_data) >= samples_per_shard or samples_processed == total_training_samples:
            if len(current_shard_data) > 0:
                shard_filename = f"flowers_train_{current_shard_idx:03d}.parquet"
                save_path = os.path.join(args.output_dir, shard_filename)
                
                print(f"Saving Shard {current_shard_idx} to {save_path} ({len(current_shard_data)} samples)...")
                
                df = pd.DataFrame(current_shard_data)
                df.to_parquet(save_path)
                
                current_shard_data = []
                current_shard_idx += 1

    print("Done. All training parquet shards generated.")

if __name__ == "__main__":
    main()