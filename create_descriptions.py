import argparse
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def setup_model(model_id, device):
    print(f"Loading VLM: {model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def load_existing_progress(output_path):
    """
    Reads the JSONL file and returns a set of filenames that have already been processed.
    """
    processed_files = set()
    if os.path.exists(output_path):
        print(f"Found existing file at {output_path}. Loading progress...")
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_files.update(entry.keys())
                except json.JSONDecodeError:
                    continue
        print(f"Resuming... {len(processed_files)} images already processed.")
    return processed_files

def generate_batch_descriptions(model, processor, images, device):
    prompt = "Describe the visual appearance of this flower briefly, focusing on colors and shapes. Do not include its name under any circumstance, only provide a visual descriptions."
    
    messages_batch = []
    for img in images:
        messages_batch.append([
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
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
            max_new_tokens=128,  # Keep it short/simple
            do_sample=False     # Deterministic
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return output_texts

def append_to_jsonl(output_path, batch_filenames, batch_descs):
    with open(output_path, 'a') as f:
        for fname, desc in zip(batch_filenames, batch_descs):
            entry = {fname: desc}
            f.write(json.dumps(entry) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate simple descriptions for Oxford Flowers")
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="generated_descriptions_7b.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_dir = os.path.join(args.data_root, "images/jpg")
    output_path = os.path.join(args.data_root, args.output_file)
    
    processed_files = load_existing_progress(output_path)
    
    all_image_files = sorted(os.listdir(images_dir))
    print(f"Total images found: {len(all_image_files)}")
    
    images_to_process = [f for f in all_image_files if f not in processed_files]
    print(f"Images remaining to process: {len(images_to_process)}")

    model, processor = setup_model(args.model_id, device)
    
    if not images_to_process:
        print("All images processed. Exiting.")
        return
    
    descriptions = {}
    
    batch_images = []
    batch_filenames = []

    for filename in tqdm(images_to_process):
        img_path = os.path.join(images_dir, filename)
        image = Image.open(img_path).convert("RGB")
        
        batch_images.append(image)
        batch_filenames.append(filename)
        
        if len(batch_images) == args.batch_size:
            batch_descs = generate_batch_descriptions(model, processor, batch_images, device)
            
            append_to_jsonl(output_path, batch_filenames, batch_descs)
            
            batch_images = []
            batch_filenames = []
                
    if batch_images:
        batch_descs = generate_batch_descriptions(model, processor, batch_images, device)
        append_to_jsonl(output_path, batch_filenames, batch_descs)

    print("Done.")

if __name__ == "__main__":
    main()
