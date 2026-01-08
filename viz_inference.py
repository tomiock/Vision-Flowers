import torch
import argparse
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from dataset import FlowersDataset

torch.multiprocessing.set_sharing_strategy('file_system')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def visualize_batch(model, processor, loader, classes, device, num_samples=9, save_path="inference_viz.png"):
    model.eval()
    
    # 1. Select random samples from the loader
    # We iterate until we collect enough samples
    collected_images = []
    collected_labels = []
    
    print(f"Collecting {num_samples} random samples...")
    all_data = list(loader)
    random.shuffle(all_data)
    
    for batch_images, _, batch_labels in all_data:
        for img, lbl in zip(batch_images, batch_labels):
            collected_images.append(img)
            collected_labels.append(lbl.item())
            if len(collected_images) >= num_samples:
                break
        if len(collected_images) >= num_samples:
            break
            
    # 2. Prepare Inference Inputs
    # We ask for a concise answer to keep it clean
    queries = ["What kind of flower is this? Answer briefly."] * len(collected_images)
    
    messages_batch = []
    for img, q in zip(collected_images, queries):
        messages_batch.append([
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}
        ])
        
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
        for msg in messages_batch
    ]
    
    # Batch process inputs
    inputs = processor(
        text=texts, images=collected_images, padding=True, return_tensors="pt"
    ).to(device)
    
    # 3. Generate
    print("Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=20,  # Keep it short for the plot title
            do_sample=False,
        )

    # 4. Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    # 5. Plotting
    # Determine grid size (approx square)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    print(f"Plotting results to {save_path}...")
    
    for i in range(num_samples):
        ax = axes[i]
        img = collected_images[i]
        pred_text = output_texts[i].strip()
        true_label_idx = collected_labels[i]
        true_label_name = classes[true_label_idx]
        
        # Check correctness (Loose match)
        is_correct = true_label_name.lower() in pred_text.lower()
        color = 'green' if is_correct else 'red'
        
        # Display Image
        ax.imshow(img)
        ax.axis('off')
        
        # Create Title
        # Truncate pred_text if too long for the plot
        
        title_text = f"True: {true_label_name}\nPred: {pred_text}"
        ax.set_title(title_text, color=color, fontsize=10, fontweight='bold')

    # Turn off remaining empty axes
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--num_samples", type=int, default=9, help="Number of images to visualize")
    parser.add_argument("--save_path", type=str, default="inference_viz.png")
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF")
    parser.add_argument("--splits_path", type=str, default="/data-net/storage2/datasets/OxfordF/my_data.json")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id}...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Dataset Setup
    imgs_path = os.path.join(args.data_root, 'images/jpg')
    labels_path = os.path.join(args.data_root, 'labels.npz')
    cat_path = os.path.join(args.data_root, 'cat_to_name.json')

    # Important: preprocess=lambda x: x gives us raw PIL images for plotting
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
        batch_size=args.num_samples, # Load enough for one batch
        shuffle=True, 
        num_workers=4, 
        collate_fn=custom_collate
    )

    # Pass dataset.classes explicitly
    visualize_batch(
        model, 
        processor, 
        val_loader, 
        dataset.classes, 
        device, 
        num_samples=args.num_samples,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()