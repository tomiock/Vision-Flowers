import torch
import argparse
import json
import os
import re
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch.distributed.checkpoint as dcp

from dataset import FlowersDataset

def evaluate_generative(model, processor, loader, classes, device):
    """
    Evaluates Top-1 accuracy by generating text and checking if the class name is present.
    """
    model.eval()
    correct = 0
    total = 0
    
    # We batch the GENERATION for speed.
    # Using tqdm(..., leave=False) to avoid cluttering the terminal output during the loop
    for images, _, labels in tqdm(loader, desc="Evaluating", leave=False):
        batch_size = len(images)
        
        # 1. Prepare Inputs
        queries = ["What type of flower is this? Answer briefly"] * batch_size
        
        messages_batch = []
        for img, q in zip(images, queries):
            messages_batch.append([
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}
            ])
            
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
            for msg in messages_batch
        ]
        
        inputs = processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(device)
        
        # 2. Fast Generation
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=20,  # Short generation = FAST
                do_sample=False     # Greedy decoding = Deterministic & Fast
            )
        
        # 3. Decode & Check
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        
        for pred_text, label_idx in zip(output_texts, labels):
            true_class_name = classes[label_idx]
            
            # LOOSE EVALUATION
            if true_class_name.lower() in pred_text.lower():
                correct += 1
            total += 1

    return correct / total

def get_checkpoint_list(root_dir):
    """
    Finds all subdirectories matching 'checkpoint-step-<number>' and returns sorted list.
    """
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    checkpoints = []
    
    for d in subdirs:
        match = re.search(r'checkpoint-step-(\d+)', d)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, os.path.join(root_dir, d)))
            
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--checkpoint_root", type=str, required=True, help="Root directory containing checkpoint-step-XXX folders")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF")
    parser.add_argument("--splits_path", type=str, default="/data-net/storage2/datasets/OxfordF/my_data.json")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Load Dataset (Once) ---
    print("Preparing Dataset...")
    imgs_path = os.path.join(args.data_root, 'images/jpg')
    labels_path = os.path.join(args.data_root, 'labels.npz')
    cat_path = os.path.join(args.data_root, 'cat_to_name.json')

    dataset = FlowersDataset(
        imgs_path, labels_path, cat_path,
        preprocess=lambda x: x,
        prompt_template="{}"
    )

    with open(args.splits_path, 'rb') as f:
        splits = json.load(f)
    test_indices = splits['trnid']
    if min(test_indices) == 1: test_indices = [x - 1 for x in test_indices]

    val_dataset = Subset(dataset, test_indices)
    
    def custom_collate(batch):
        images = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        labels = torch.tensor([item[2] for item in batch])
        return images, texts, labels

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        collate_fn=custom_collate
    )

    # --- 2. Initialize Model Architecture (Once) ---
    print(f"Initializing Model Architecture: {args.model_id}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cpu" 
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # --- 3. Iterate Checkpoints ---
    checkpoints = get_checkpoint_list(args.checkpoint_root)
    if not checkpoints:
        print(f"No 'checkpoint-step-XXX' folders found in {args.checkpoint_root}")
        return

    print(f"Found {len(checkpoints)} checkpoints to evaluate: {[s for s, _ in checkpoints]}")
    
    results = []

    for step, cp_path in checkpoints:
        print(f"\n--- Evaluating Step {step} ---")
        print(f"Loading weights from {cp_path}...")
        
        # Container must match training save structure: {"model": self.model, ...}
        loading_state_dict = {"model": model}
        
        try:
            # Load weights in-place (CPU first is safer for DCP)
            dcp.load(
                state_dict=loading_state_dict,
                checkpoint_id=cp_path,
            )
        except Exception as e:
            print(f"FAILED to load step {step}: {e}")
            continue

        # Move to GPU for inference
        model.to(device)
        
        # Evaluate
        acc = evaluate_generative(model, processor, val_loader, dataset.classes, device)
        print(f"Step {step} Accuracy: {acc:.4f}")
        
        results.append({"Step": step, "Accuracy": acc})
        
        # Move back to CPU to clear VRAM for next load (optional but safe)
        model.cpu()
        torch.cuda.empty_cache()

    # --- 4. Final Report ---
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    
    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
        
        # Optional: Save CSV
        save_csv_path = os.path.join(args.checkpoint_root, "eval_results.csv")
        df.to_csv(save_csv_path, index=False)
        print(f"\nResults saved to {save_csv_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()