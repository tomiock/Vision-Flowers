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
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import clip

# --- Custom Dataset for Descriptions ---
class FlowersDescriptionDataset(Dataset):
    def __init__(self, images_dir, descriptions_path, preprocess):
        """
        Args:
            images_dir: Path to image folder.
            descriptions_path: Path to the JSONL file created by generate_descriptions.py.
            preprocess: CLIP preprocessing function.
        """
        self.images_dir = images_dir
        self.preprocess = preprocess
        
        # Load descriptions from JSONL
        self.descriptions_map = {}
        print(f"Loading descriptions from {descriptions_path}...")
        
        with open(descriptions_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Each line is { "filename": "description" }
                    self.descriptions_map.update(entry)
                except json.JSONDecodeError:
                    continue
            
        # Ensure we only use images that exist in both folder and json
        self.image_filenames = sorted(list(self.descriptions_map.keys()))
        
        # Filter strictly for existing images
        self.image_filenames = [
            f for f in self.image_filenames 
            if os.path.exists(os.path.join(images_dir, f))
        ]
        
        print(f"Dataset loaded: {len(self.image_filenames)} valid image-text pairs.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, filename)
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)
        
        # Load Description
        text = self.descriptions_map[filename]
        # Tokenize for CLIP
        text_token = clip.tokenize(text, truncate=True)[0]
        
        return image_tensor, text_token, filename

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def configure_model_freezing(model, mode):
    """
    Configures which parts of the model are frozen/trainable.
    """
    print(f"\nConfiguring Model Freezing: Mode = {mode}")
    
    # Reset all parameters to trainable first
    for param in model.parameters():
        param.requires_grad = True

    if mode == "text_only":
        # Freeze visual encoder
        for param in model.visual.parameters():
            param.requires_grad = False
        print(" -> Vision Encoder FROZEN. Training Text Encoder only.")

    elif mode == "vision_only":
        # Freeze text transformer
        for param in model.transformer.parameters():
            param.requires_grad = False
        for param in model.token_embedding.parameters():
            param.requires_grad = False
        for param in model.ln_final.parameters():
            param.requires_grad = False
        if hasattr(model, 'text_projection'):
             model.text_projection.requires_grad = False
        print(" -> Text Encoder FROZEN. Training Vision Encoder only.")

    elif mode == "early_layers":
        # Strategy: Freeze EVERYTHING first, then UNFREEZE the last 2 layers + Heads
        
        # 1. Freeze everything
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. Unfreeze Vision: Last 2 ResBlocks + Final Norm + Projection
        # Unfreeze last 2 layers of Vision Transformer
        num_visual_layers = len(model.visual.transformer.resblocks)
        for i, block in enumerate(model.visual.transformer.resblocks):
            if i >= num_visual_layers - 2:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Unfreeze Vision Final Norm
        for param in model.visual.ln_post.parameters():
            param.requires_grad = True
            
        # Unfreeze Vision Projection (if it exists as a parameter)
        if hasattr(model.visual, 'proj') and model.visual.proj is not None:
            # model.visual.proj is a nn.Parameter in CLIP
            model.visual.proj.requires_grad = True

        # 3. Unfreeze Text: Last 2 ResBlocks + Final Norm + Projection
        num_text_layers = len(model.transformer.resblocks)
        for i, block in enumerate(model.transformer.resblocks):
            if i >= num_text_layers - 2:
                for param in block.parameters():
                    param.requires_grad = True
                    
        # Unfreeze Text Final Norm
        for param in model.ln_final.parameters():
            param.requires_grad = True
            
        # Unfreeze Text Projection
        if hasattr(model, 'text_projection') and model.text_projection is not None:
            model.text_projection.requires_grad = True

        print(" -> All parameters FROZEN except the LAST 2 LAYERS and FINAL PROJECTIONS of both encoders.")

    elif mode == "none":
        print(" -> All parameters TRAINABLE.")
        
    else:
        raise ValueError(f"Unknown freeze mode: {mode}")

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" -> Trainable Params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return trainable_params

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    loss_fct = nn.CrossEntropyLoss()
    
    for images, texts, _ in loader:
        images, texts = images.to(device), texts.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)
        
        # Contrastive Loss: match image[i] with text[i]
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        loss = (loss_fct(logits_per_image, ground_truth) + 
                loss_fct(logits_per_text, ground_truth)) / 2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        
        wandb.log({"train/batch_loss": loss.item()})
        
    return total_loss / len(loader)

def evaluate_retrieval(model, loader, device):
    """
    Evaluates Text-to-Image Retrieval (Recall@K).
    """
    model.eval()
    print("Computing embeddings for retrieval evaluation...")
    
    all_image_feats = []
    all_text_feats = []
    
    with torch.no_grad():
        for images, texts, _ in loader:
            images = images.to(device)
            texts = texts.to(device)
            
            # Encode & Normalize
            img_f = model.encode_image(images)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            
            txt_f = model.encode_text(texts)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            
            all_image_feats.append(img_f)
            all_text_feats.append(txt_f)
            
    all_image_feats = torch.cat(all_image_feats, dim=0)
    all_text_feats = torch.cat(all_text_feats, dim=0)
    
    n_samples = all_text_feats.shape[0]
    
    print(f"Calculating similarity matrix ({n_samples}x{n_samples})...")
    sim_matrix = all_text_feats @ all_image_feats.T
    
    ranks = torch.argsort(sim_matrix, dim=1, descending=True)
    ground_truth = torch.arange(n_samples, device=device).view(-1, 1)
    
    hits_r1 = (ranks[:, :1] == ground_truth).any(dim=1).float().mean().item()
    hits_r5 = (ranks[:, :5] == ground_truth).any(dim=1).float().mean().item()
    hits_r10 = (ranks[:, :10] == ground_truth).any(dim=1).float().mean().item()
    
    print(f"R@1: {hits_r1:.4f} | R@5: {hits_r5:.4f} | R@10: {hits_r10:.4f}")
    return hits_r1, hits_r5, hits_r10

def main():
    parser = argparse.ArgumentParser(description="Train CLIP for Image Retrieval (Description matching)")
    
    # Same config as train.py
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--architecture", type=str, default="ViT-B/32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_mode", type=str, default="none", 
                        choices=["text_only", "vision_only", "early_layers", "none"],
                        help="Select which parts of the model to train.")
    
    # Paths
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF")
    # Updated default to .jsonl
    parser.add_argument("--descriptions_file", type=str, default="/data-net/storage2/datasets/OxfordF/generated_descriptions_7b.json")
    parser.add_argument("--splits_path", type=str, default="/data-net/storage2/datasets/OxfordF/my_data.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_retrieval")

    # Metadata
    parser.add_argument("--project_name", type=str, default="clip-flowers-retrieval")
    parser.add_argument("--entity_name", type=str, default="uab-deeplearning-2025")

    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize W&B
    wandb.init(
        name=f"retrieval-{args.architecture}-{args.freeze_mode}",
        entity=args.entity_name,
        project=args.project_name,
        config=vars(args),
        tags=["retrieval", "finetune", args.freeze_mode]
    )

    # Load Model
    model, preprocess = clip.load(args.architecture, device=device, jit=False)
    model = model.float()

    # Apply Freezing Logic
    trainable_params_count = configure_model_freezing(model, args.freeze_mode)
    wandb.config.update({"trainable_params": trainable_params_count})

    # Dataset Setup
    images_dir = os.path.join(args.data_root, 'images/jpg')
    desc_path = os.path.join(args.data_root, args.descriptions_file)
    
    if not os.path.exists(desc_path):
        raise FileNotFoundError(f"Descriptions file not found at {desc_path}. Run generate_descriptions.py first.")

    dataset = FlowersDescriptionDataset(images_dir, desc_path, preprocess)

    # Load Splits
    with open(args.splits_path, 'rb') as f:
        splits = json.load(f)

    # --- Same split logic as train.py ---
    train_indices = splits['valid'] + splits['tstid']
    test_indices = splits['trnid']
    
    if min(train_indices) == 1 or min(test_indices) == 1:
        train_indices = [x - 1 for x in train_indices]
        test_indices = [x - 1 for x in test_indices]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Retrieval Training: {len(train_dataset)} pairs")
    print(f"Retrieval Evaluation: {len(val_dataset)} pairs")

    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr, weight_decay=args.wd)

    best_r1 = 0.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Evaluate Retrieval on Test set
        r1, r5, r10 = evaluate_retrieval(model, val_loader, device)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | R@1: {r1:.4f}")
        
        wandb.log({
            "train/loss": train_loss,
            "eval/R@1": r1,
            "eval/R@5": r5,
            "eval/R@10": r10,
            "epoch": epoch + 1
        })
        
        if r1 > best_r1:
            best_r1 = r1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_retrieval_model.pt"))

    print("Training Complete.")

if __name__ == "__main__":
    main()
