import os
import json
import wandb
import torch
import argparse
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import clip
from dataset import FlowersDataset

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_zero_shot(model, loader, classes, prompt_template, device):
    """
    Exact same evaluation function as train.py
    """
    model.eval()
    print("Building Zero-shot Classifier...")
    
    # Pre-compute text features for all classes
    text_inputs = torch.cat([clip.tokenize(prompt_template.format(c)) for c in classes]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, _, labels in loader:
            images = images.to(device)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Similarity calculation
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity.topk(1)
            
            all_preds.extend(indices.squeeze().cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

def main():
    parser = argparse.ArgumentParser(description="Baseline Zero-Shot Evaluation (No Training)")
    
    # Config arguments (kept same as train.py for consistency in WandB)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--architecture", type=str, default="ViT-B/32", help="CLIP Architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt_template", type=str, default="an image of the {} flower", help="Prompt template")
    
    # Paths
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF", help="Dataset root")
    parser.add_argument("--splits_path", type=str, default="/data-net/storage2/datasets/OxfordF/my_data.json", help="Splits JSON path")

    # Metadata
    parser.add_argument("--project_name", type=str, default="clip-flowers-finetune")
    parser.add_argument("--entity_name", type=str, default="uab-deeplearning-2025")

    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Init WandB
    wandb.init(
        name=f"baseline-{args.architecture}-zero-shot",
        entity=args.entity_name,
        project=args.project_name,
        config=vars(args),
        tags=["baseline", "zero-shot"]
    )

    # Load Model (Vanilla CLIP)
    print(f"Loading vanilla {args.architecture} model...")
    model, preprocess = clip.load(args.architecture, device=device, jit=False)
    
    # CRITICAL: Force Float32 to match train.py behavior exactly
    model = model.float()

    # Prepare Data
    imgs_path = os.path.join(args.data_root, 'images/jpg')
    labels_path = os.path.join(args.data_root, 'labels.npz')
    cat_path = os.path.join(args.data_root, 'cat_to_name.json')

    dataset = FlowersDataset(
        imgs_path, 
        labels_path, 
        cat_path,
        preprocess=preprocess, 
        prompt_template=args.prompt_template
    )

    with open(args.splits_path, 'rb') as f:
        splits = json.load(f)

    # We only care about the validation/test split for baseline eval
    test_indices = splits['trnid'] # Based on your previous code where 'trnid' was used for validation
    
    # Fix 1-based indexing if present
    if min(test_indices) == 1:
        test_indices = [x - 1 for x in test_indices]

    val_dataset = Subset(dataset, test_indices)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Running Baseline Evaluation on {len(val_dataset)} images...")

    # Run Evaluation
    y_true, y_pred = evaluate_zero_shot(model, val_loader, dataset.classes, args.prompt_template, device)

    # Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Baseline Accuracy: {acc:.4f}")
    
    # Log to WandB
    wandb.log({
        "eval/accuracy": acc,
        "eval/f1_weighted": f1,
        "eval/precision_weighted": precision,
        "eval/recall_weighted": recall
    })
    
    # detailed report
    report = classification_report(y_true, y_pred, target_names=dataset.classes, output_dict=True)
    wandb.log({"eval/classification_report": wandb.Table(dataframe=pd.DataFrame(report).transpose())})
    
    print("Baseline evaluation complete. Results logged to WandB.")

if __name__ == "__main__":
    main()