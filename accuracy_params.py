import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def get_model_params(arch_name):
    """
    Returns approx parameter count (in Millions) for standard CLIP models.
    """
    params_map = {
        "RN50": 102.01,
        "RN101": 119.69,
        "RN50x4": 178.30,
        "RN50x16": 290.98,
        "RN50x64": 623.26,
        "ViT-B/32": 151.28,
        "ViT-B/16": 149.62,
        "ViT-L/14": 427.62,
        "ViT-L/14@336px": 427.94
    }
    return params_map.get(arch_name, None)

def get_model_family(arch_name):
    """
    Classify architecture into 'ResNet' or 'ViT'.
    """
    if "RN" in arch_name:
        return "ResNet"
    elif "ViT" in arch_name:
        return "ViT"
    else:
        return "Other"

def main():
    parser = argparse.ArgumentParser(description="Plot Accuracy vs Parameters for CLIP Architecture Sweep")
    parser.add_argument("--entity", type=str, default="uab-deeplearning-2025", help="WandB Entity")
    parser.add_argument("--project", type=str, default="clip-flowers-architecture-sweep", help="WandB Project Name")
    args = parser.parse_args()

    api = wandb.Api()
    
    # Fetch runs
    print(f"Fetching runs from {args.entity}/{args.project}...")
    runs = api.runs(f"{args.entity}/{args.project}")

    data = []
    for run in runs:
        # Get config
        arch = run.config.get("architecture", "Unknown")
        
        # Get Summary Metrics (Final results)
        summary = run.summary
        acc1 = summary.get("eval/top1", None)
        
        # Get parameter count from map
        params_m = get_model_params(arch)
        
        # Determine family
        family = get_model_family(arch)

        if acc1 is not None and params_m is not None:
            data.append({
                "Architecture": arch,
                "Params (M)": params_m,
                "Top-1 Accuracy": acc1,
                "Family": family
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No valid data found (check if runs have 'eval/top1' and valid architectures).")
        return

    # Sort for cleaner plotting
    df = df.sort_values("Params (M)")

    print(f"Found {len(df)} runs. Generating plot...")

    # Set up the plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Create Scatter Plot with distinct colors for families
    # Using 'hue' for color differentiation and 'style' for marker shape as well
    sns.scatterplot(
        data=df, 
        x="Params (M)", 
        y="Top-1 Accuracy", 
        hue="Family", 
        style="Family",
        s=150, # Marker size
        palette={"ResNet": "tab:blue", "ViT": "tab:orange", "Other": "gray"}
    )

    # Annotate points with model names
    for i in range(df.shape[0]):
        plt.text(
            df["Params (M)"].iloc[i] + 5, # Offset x slightly
            df["Top-1 Accuracy"].iloc[i], 
            df["Architecture"].iloc[i], 
            fontsize=9,
            va='center'
        )

    plt.title("Accuracy across different Vision Encoder architectures", fontsize=14)
    plt.xlabel("Number of Parameters (Millions)", fontsize=12)
    plt.ylabel("Top-1 Accuracy", fontsize=12)
    plt.xscale("log")
    plt.legend(title="Architecture Family", loc="lower right")
    
    plt.tight_layout()
    output_filename = "accuracy_vs_params.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
