import clip
import torch
import pandas as pd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    available_models = clip.available_models()
    
    print(f"Checking parameters for: {available_models}")
    print(f"Using device: {device}\n")

    results = []

    for model_name in available_models:
        try:
            print(f"Loading {model_name}...", end=" ", flush=True)
            model, _ = clip.load(model_name, device=device, jit=False)
            
            # Calculate parameters in millions
            params = count_parameters(model)
            params_m = params / 1_000_000
            
            results.append({
                "Model": model_name,
                "Parameters": params,
                "Params (M)": f"{params_m:.2f} M"
            })
            
            print(f"Done. ({params_m:.2f} M)")
            
            # Clear memory to prevent OOM on next iteration
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed: {e}")

    # Sort by parameter count descending
    df = pd.DataFrame(results).sort_values(by="Parameters", ascending=False)
    
    print("\n--- Model Parameter Counts ---")
    print(df[["Model", "Params (M)"]].to_string(index=False))

if __name__ == "__main__":
    main()