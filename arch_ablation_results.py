import wandb
import pandas as pd
import argparse

def get_model_params(arch_name):
    """
    Returns approx parameter count (in Millions) for standard CLIP models.
    Source: OpenAI CLIP paper / OpenCLIP config
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
    return params_map.get(arch_name, "N/A")

def main():
    parser = argparse.ArgumentParser(description="Extract WandB results for CLIP Architecture Sweep")
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
        batch_size = run.config.get("batch_size", "Unknown")
        
        # Get Summary Metrics (Final results)
        summary = run.summary
        
        # Extract metrics (handling potential missing keys safely)
        acc1 = summary.get("eval/top1", 0.0)
        acc3 = summary.get("eval/top3", 0.0)
        acc5 = summary.get("eval/top5", 0.0)
        precision_weighted = summary.get("eval/precision_weighted", 0.0)
        recall_weighted = summary.get("eval/recall_weighted", 0.0)

        # Get parameter count from map
        params_m = get_model_params(arch)

        data.append({
            "Model Name": arch,
            "Params (M)": params_m,
            "Top-1 Acc": acc1,
            "Top-3 Acc": acc3,
            "Top-5 Acc": acc5,
            "Weighted Precision": precision_weighted,
            "Weighted Recall": recall_weighted,
            "Batch Size": batch_size
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    print(df)
    
    # Sort by Params or Accuracy as desired (e.g., by Params)
    # Since Params might be strings ("N/A"), we need to be careful, but assuming standard models:
    df["Params (M)"] = pd.to_numeric(df["Params (M)"], errors='coerce')
    df = df.sort_values("Params (M)")

    print(f"Found {len(df)} runs.")
    print(df)

    # --- Generate LaTeX Table ---
    print("\nGenerating LaTeX Table...")
    
    latex_str = "\\begin{table}[ht]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Zero-Shot Classification Results on Oxford Flowers Dataset across CLIP Architectures}\n"
    latex_str += "\\label{tab:clip_results}\n"
    latex_str += "\\resizebox{\\textwidth}{!}{%\n"
    latex_str += "\\begin{tabular}{l c c c c c c c}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\textbf{Model} & \\textbf{Params (M)} & \\textbf{Batch Size} & \\textbf{Top-1} & \\textbf{Top-3} & \\textbf{Top-5} & \\textbf{W. Precision} & \\textbf{W. Recall} \\\\\n"
    latex_str += "\\midrule\n"

    for _, row in df.iterrows():
        # Formatting metrics to 4 decimal places or percentages as preferred
        # Assuming metrics are 0.0-1.0, converting to percentage for readability usually looks better in tables, 
        # but user asked for raw values likely. Let's stick to 4 decimals.
        
        line = f"{row['Model Name']} & "
        line += f"{row['Params (M)']:.2f} & "
        line += f"{row['Batch Size']} & "
        line += f"{row['Top-1 Acc']:.4f} & "
        line += f"{row['Top-3 Acc']:.4f} & "
        line += f"{row['Top-5 Acc']:.4f} & "
        line += f"{row['Weighted Precision']:.4f} & "
        line += f"{row['Weighted Recall']:.4f} \\\\"
        latex_str += line + "\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}}\n"
    latex_str += "\\end{table}"

    print(latex_str)

    # Save to file
    with open("clip_results_table.tex", "w") as f:
        f.write(latex_str)
    print("\nSaved LaTeX table to clip_results_table.tex")

if __name__ == "__main__":
    main()
