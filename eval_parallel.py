import os
import json
import wandb
import torch
import argparse
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from dataset import FlowersDataset

CLASS_BATCH_SIZE = 32

def setup_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)

def get_log_likelihood_batch(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs["labels"][:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        token_losses = token_losses.view(labels.shape)
        return -token_losses.sum(dim=1)

def evaluate_single_image(model, processor, image, classes, device, prompt_template):
    """
    Correctly constructs inputs by concatenating [Image+Query] prefix with [Class Name] suffix.
    """
    # 1. PREPARE THE PREFIX (Image + User Query) ONCE
    # This ensures input_ids contains the correct image placeholders.
    prefix_msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_template}
            ]
        }
    ]
    
    # Generate text for the prefix (ends with "Assistant:")
    prefix_text = processor.apply_chat_template(prefix_msg, tokenize=False, add_generation_prompt=True)
    
    # Run processor on the image to get the "heavy" tensors and correct IDs
    prefix_inputs = processor(
        text=[prefix_text],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt"
    )
    
    # Extract the base components
    # prefix_input_ids shape: [1, Seq_Len_with_Image_Tokens]
    prefix_input_ids = prefix_inputs.input_ids[0].to(device) 
    
    # Vision tensors
    pixel_values = prefix_inputs.pixel_values.to(device, dtype=model.dtype)
    image_grid_thw = prefix_inputs.image_grid_thw.to(device)

    all_scores = []

    # 2. ITERATE CLASSES IN CHUNKS
    for i in range(0, len(classes), CLASS_BATCH_SIZE):
        batch_classes = classes[i : i + CLASS_BATCH_SIZE]
        current_batch_size = len(batch_classes)
        
        # Lists to build the batch
        batch_input_ids = []
        batch_labels = []
        
        for cls_name in batch_classes:
            # Tokenize JUST the class name. 
            # Note: We use add_special_tokens=False to avoid adding extra BOS/EOS in the middle
            suffix_ids = processor.tokenizer(cls_name, add_special_tokens=False).input_ids
            suffix_tensor = torch.tensor(suffix_ids, device=device)
            
            # Concatenate [Prefix] + [Suffix]
            full_seq = torch.cat([prefix_input_ids, suffix_tensor])
            batch_input_ids.append(full_seq)
            
            # Create Labels (Mask the prefix with -100)
            labels = torch.full_like(full_seq, -100)
            # Only calculate loss on the suffix part
            labels[len(prefix_input_ids):] = suffix_tensor
            batch_labels.append(labels)
            
        # 3. PAD THE BATCH
        # Manually pad sequences to the longest in this batch
        max_len = max(len(x) for x in batch_input_ids)
        pad_id = processor.tokenizer.pad_token_id
        
        final_input_ids = torch.full((current_batch_size, max_len), pad_id, dtype=torch.long, device=device)
        final_labels = torch.full((current_batch_size, max_len), -100, dtype=torch.long, device=device)
        final_attention = torch.zeros((current_batch_size, max_len), dtype=torch.long, device=device)
        
        for idx, (seq, lbl) in enumerate(zip(batch_input_ids, batch_labels)):
            l = len(seq)
            final_input_ids[idx, :l] = seq
            final_labels[idx, :l] = lbl
            final_attention[idx, :l] = 1 # 1 for valid tokens, 0 for pad
            
        # 4. EXPAND VISION TENSORS
        # Repeat the visual features 'current_batch_size' times.
        # This matches the batch size of the text inputs.
        batch_pixel_values = pixel_values.repeat(current_batch_size, 1)
        batch_image_grid_thw = image_grid_thw.repeat(current_batch_size, 1)
        
        # 5. FORWARD PASS
        model_inputs = {
            "input_ids": final_input_ids,
            "attention_mask": final_attention,
            "labels": final_labels,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw
        }
        
        batch_scores = get_log_likelihood_batch(model, model_inputs)
        all_scores.extend(batch_scores.float().cpu().numpy())
        
    return np.array(all_scores)

def worker_fn(rank, world_size, args, shared_results):
    setup_process(rank, world_size)
    device = f"cuda:{rank}"
    
    print(f"[GPU {rank}] Loading Model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    # Dataset Setup
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

    # Distributed Split
    my_indices = test_indices[rank::world_size]
    
    subset = Subset(dataset, my_indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    print(f"[GPU {rank}] Processing {len(my_indices)} images...")
    
    local_hits_top1 = []
    local_hits_top3 = []
    local_hits_top5 = []
    
    query_text = "Identify the specific type of this flower."

    for img, _, label_idx in tqdm(loader, position=rank, desc=f"GPU {rank}"):
        true_label = int(label_idx)
        
        scores = evaluate_single_image(
            model, processor, img, dataset.classes, device, query_text
        )
        
        predicted_indices = np.argsort(scores)[::-1]
        
        local_hits_top1.append(true_label in predicted_indices[:1])
        local_hits_top3.append(true_label in predicted_indices[:3])
        local_hits_top5.append(true_label in predicted_indices[:5])

    # Save results
    shared_results[rank] = (
        len(local_hits_top1),
        sum(local_hits_top1),
        sum(local_hits_top3),
        sum(local_hits_top5)
    )
    print(f"[GPU {rank}] Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF")
    parser.add_argument("--splits_path", type=str, default="/data-net/storage2/datasets/OxfordF/my_data.json")
    parser.add_argument("--project_name", type=str, default="clip-flowers-finetune")
    parser.add_argument("--entity_name", type=str, default="uab-deeplearning-2025")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"--- Parallel Eval on {world_size} GPUs ---")
    
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    shared_results = manager.dict()

    mp.spawn(worker_fn, args=(world_size, args, shared_results), nprocs=world_size)

    print("\n--- Aggregating Results ---")
    total_samples = 0
    total_t1 = 0
    total_t3 = 0
    total_t5 = 0

    for rank in range(world_size):
        count, s1, s3, s5 = shared_results[rank]
        total_samples += count
        total_t1 += s1
        total_t3 += s3
        total_t5 += s5

    acc_t1 = total_t1 / total_samples
    acc_t3 = total_t3 / total_samples
    acc_t5 = total_t5 / total_samples

    print(f"Top-1: {acc_t1:.4f}")
    print(f"Top-3: {acc_t3:.4f}")
    print(f"Top-5: {acc_t5:.4f}")

    wandb.init(
        name=f"baseline-{args.model_id.split('/')[-1]}-parallel",
        entity=args.entity_name,
        project=args.project_name,
        config=vars(args)
    )
    wandb.log({"eval/top1": acc_t1, "eval/top3": acc_t3, "eval/top5": acc_t5})

if __name__ == "__main__":
    main()
