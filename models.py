import clip
import torch
import pandas as pd

def inspect_vision_encoder(model):
    """
    Inspects the vision encoder to count layers and determine width.
    """
    visual = model.visual
    
    # 1. Vision Transformer (ViT)
    if hasattr(visual, 'transformer'):
        # For ViT, the transformer is usually in visual.transformer.resblocks
        layer_count = len(visual.transformer.resblocks)
        width = visual.transformer.width
        return "ViT", layer_count, width
    
    # 2. ResNet (ModifiedResNet in CLIP)
    elif hasattr(visual, 'layer1'):
        count = 0
        count += len(visual.layer1)
        count += len(visual.layer2)
        count += len(visual.layer3)
        count += len(visual.layer4)
        
        # Determine width from the last layer's output channels
        # Typically visual.layer4[-1] is a Bottleneck
        last_block = visual.layer4[-1]
        
        # The last conv in a Bottleneck is usually `conv3`
        if hasattr(last_block, 'conv3'):
            final_width = last_block.conv3.out_channels
        else:
            final_width = "Unknown"

        return "ResNet", count, final_width
        
    return "Unknown", 0, 0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # List of models from the table to verify
    models_to_verify = [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px"
    ]
    
    print(f"Verifying Vision Encoder depths and widths on {device}...\n")
    
    results = []
    
    # Standard ResNet50 final width is 2048
    BASE_RN50_WIDTH = 2048

    for model_name in models_to_verify:
        try:
            print(f"Loading {model_name}...", end=" ", flush=True)
            # Load model (cpu is fine for structural inspection)
            model, _ = clip.load(model_name, device=device, jit=False)
            
            enc_type, layer_count, width = inspect_vision_encoder(model)
            
            # Calculate Multiplier for ResNets
            multiplier = "N/A"
            if enc_type == "ResNet" and isinstance(width, int):
                mult = width / BASE_RN50_WIDTH
                multiplier = f"{mult:.0f}x"
            
            results.append({
                "Model Name": model_name,
                "Encoder Type": enc_type,
                "Layer/Block Count": layer_count,
                "Width": width,
                "ResNet Mult": multiplier
            })
            
            print(f"Done. -> {layer_count} layers, width {width}")
            
            # Cleanup to save memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                "Model Name": model_name,
                "Encoder Type": "Error",
                "Layer/Block Count": "N/A",
                "Width": "N/A",
                "ResNet Mult": "N/A"
            })

    # Display results
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save to CSV for record
    df.to_csv("verified_model_stats.csv", index=False)
    print("\nResults saved to verified_model_stats.csv")

if __name__ == "__main__":
    main()
