import clip
import torch
import pandas as pd

def inspect_vision_encoder(model):
    """
    Inspects the vision encoder to count layers and determine widths.
    """
    visual = model.visual
    
    # 1. Vision Transformer (ViT)
    if hasattr(visual, 'transformer'):
        layer_count = len(visual.transformer.resblocks)
        width = visual.transformer.width
        return "ViT", layer_count, width, "N/A"
    
    # 2. ResNet (ModifiedResNet in CLIP)
    elif hasattr(visual, 'layer1'):
        count = 0
        count += len(visual.layer1)
        count += len(visual.layer2)
        count += len(visual.layer3)
        count += len(visual.layer4)
        
        # Get Final Output Width
        last_block = visual.layer4[-1]
        if hasattr(last_block, 'conv3'):
            final_width = last_block.conv3.out_channels
        else:
            final_width = "Unknown"

        # Get Base/Stem Width to verify multiplier
        # ModifiedResNet usually has conv1
        if hasattr(visual, 'conv1'):
            stem_width = visual.conv1.out_channels
        else:
            stem_width = "Unknown"

        return "ResNet", count, final_width, stem_width
        
    return "Unknown", 0, 0, 0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # List of models from the table to verify
    models_to_verify = [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
    ]
    
    print(f"Verifying ResNet Widths on {device}...\n")
    
    results = []
    
    # Standard ResNet50 stem width is usually 64
    BASE_RN50_STEM = 64

    for model_name in models_to_verify:
        try:
            print(f"Loading {model_name}...", end=" ", flush=True)
            model, _ = clip.load(model_name, device=device, jit=False)
            
            enc_type, layer_count, final_width, stem_width = inspect_vision_encoder(model)
            
            # Calculate Multiplier based on Stem
            stem_mult = "N/A"
            if isinstance(stem_width, int):
                mult = stem_width / BASE_RN50_STEM
                stem_mult = f"{mult:.0f}x"
            
            results.append({
                "Model": model_name,
                "Layers": layer_count,
                "Stem Width": stem_width,
                "Stem Mult": stem_mult,
                "Final Width": final_width
            })
            
            print(f"Done. Stem: {stem_width} ({stem_mult})")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed: {e}")

    # Display results
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("RESNET WIDTH ANALYSIS")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv("resnet_width_analysis.csv", index=False)

if __name__ == "__main__":
    main()
