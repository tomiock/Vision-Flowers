import os
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
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

def save_checkpoint(model, optimizer, epoch, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    for step, (images, texts, _) in enumerate(loader):
        images, texts = images.to(device), texts.to(device)
        
        optimizer.zero_grad()
        
        logits_per_image, logits_per_text = model(images, texts)
        
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        
        loss = (loss_img(logits_per_image, ground_truth) + 
                loss_txt(logits_per_text, ground_truth)) / 2
        
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

        wandb.log({"train/batch_loss": batch_loss, "train/lr": optimizer.param_groups[0]['lr']})
        
    return total_loss / len(loader)

def validate_loss(model, loader, device):
    model.eval()
    total_loss = 0
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, texts, _ in loader:
            images, texts = images.to(device), texts.to(device)
            
            logits_per_image, logits_per_text = model(images, texts)
            
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = (loss_img(logits_per_image, ground_truth) + 
                    loss_txt(logits_per_text, ground_truth)) / 2
            
            total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_zero_shot(model, loader, classes, prompt_template, device):
    """
    Evaluates Top-1, Top-3, and Top-5 accuracy.
    Returns: (y_true, y_pred_top1, acc1, acc3, acc5)
    """
    model.eval()
    print("Building Zero-shot Classifier...")
    
    text_inputs = torch.cat([clip.tokenize(prompt_template.format(c)) for c in classes]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    all_k_preds = [] # Will store indices (N, 5)
    all_labels = []
    
    print(f"Evaluating on {len(loader.dataset)} images...")

    with torch.no_grad():
        for images, _, labels in loader:
            images = images.to(device)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            values, indices = similarity.topk(5) 
            
            all_k_preds.append(indices.cpu())
            all_labels.append(labels)

    all_k_preds = torch.cat(all_k_preds, dim=0) 
    all_labels = torch.cat(all_labels, dim=0)
    
    ground_truth = all_labels.view(-1, 1)
    
    matches = (all_k_preds == ground_truth)
    
    top1 = matches[:, :1].any(dim=1).float().mean().item()
    top3 = matches[:, :3].any(dim=1).float().mean().item()
    top5 = matches[:, :5].any(dim=1).float().mean().item()
    
    print(f"Top-1: {top1:.4f} | Top-3: {top3:.4f} | Top-5: {top5:.4f}")
    
    # Return 1D arrays for sklearn metrics (just top-1) AND the scalar accuracies
    return all_labels.numpy(), all_k_preds[:, 0].numpy(), top1, top3, top5

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Oxford Flowers (Float32 + TopK)")
    
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam Beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam Beta2")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam Epsilon")
    parser.add_argument("--architecture", type=str, default="ViT-B/32", help="CLIP Architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt_template", type=str, default="an image of the {} flower", help="Prompt template")
    
    # Paths
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF", help="Dataset root")
    parser.add_argument("--splits_path", type=str, default="/data-net/storage2/datasets/OxfordF/my_data.json", help="Splits JSON path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save models")

    # Metadata
    parser.add_argument("--project_name", type=str, default="clip-flowers-finetune")
    parser.add_argument("--entity_name", type=str, default="uab-deeplearning-2025")

    args = parser.parse_args()
    
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        name=f"run-{args.architecture}-lr{args.lr}-bs{args.batch_size}",
        entity=args.entity_name,
        project=args.project_name,
        config=vars(args)
    )

    model, preprocess = clip.load(args.architecture, device=device, jit=False) 
    
    model = model.float()

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

    train_indices = splits['valid'] + splits['tstid']
    test_indices = splits['trnid']
    
    #train_indices = splits['trnid']
    #test_indices = splits['tstid'] + splits['valid']
    
    # Fix 1-based indexing if present
    if min(train_indices) == 1 or min(test_indices) == 1:
        train_indices = [x - 1 for x in train_indices]
        test_indices = [x - 1 for x in test_indices]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2), 
        eps=args.eps, 
        weight_decay=args.wd
    )

    print(f"Starting training (Float32): {len(train_dataset)} train, {len(val_dataset)} val")
    
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_loss(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "epoch": epoch + 1,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, save_path)
            wandb.run.summary["best_val_loss"] = best_val_loss

    print("\nTraining complete. Running Evaluation on Best Model...")
    
    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"))
    model.load_state_dict(checkpoint['state_dict'])

    y_true, y_pred, acc1, acc3, acc5 = evaluate_zero_shot(model, val_loader, dataset.classes, args.prompt_template, device)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Final Accuracy (Top-1): {acc1:.4f}")
    
    wandb.log({
        "eval/accuracy": acc1,
        "eval/top1": acc1,
        "eval/top3": acc3,
        "eval/top5": acc5,
        "eval/f1_weighted": f1,
        "eval/precision_weighted": precision,
        "eval/recall_weighted": recall
    })
    
    report = classification_report(y_true, y_pred, target_names=dataset.classes, output_dict=True)
    wandb.log({"eval/classification_report": wandb.Table(dataframe=pd.DataFrame(report).transpose())})

if __name__ == "__main__":
    main()