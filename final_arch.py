import wandb
import pandas as pd
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
    return params_map.get(arch_name, "N/A")

def main():
    parser = argparse.ArgumentParser(description="Extract results for runs with a specific tag")
    parser.add_argument("--entity", type=str, default="uab-deeplearning-2025", help="WandB Entity")
    parser.add_argument("--project", type=str, default="clip-flowers-finetune", help="WandB Project Name")
    parser.add_argument("--tag", type=str, default="final_arch", help="Tag to filter runs by")
    args = parser.parse_args()

    api = wandb.Api()
    
    # Fetch runs from the specific project
    print(f"Fetching runs from {args.entity}/{args.project}...")
    runs = api.runs(f"{args.entity}/{args.project}")

    data = []
    
    print(f"Filtering for runs with tag: '{args.tag}'...")
    
    for run in runs:
        # Check if the tag exists in the run's tag list
        if args.tag in run.tags:
            
            # Get config
            arch = run.config.get("architecture", "Unknown")
            batch_size = run.config.get("batch_size", "Unknown")

            params = run.config.get("trainable_params", "UNK")
            
            summary = run.summary
            
            acc1 = summary.get("eval/top1", 0.0)
            acc3 = summary.get("eval/top3", 0.0)
            acc5 = summary.get("eval/top5", 0.0)
            precision_weighted = summary.get("eval/precision_weighted", 0.0)
            recall_weighted = summary.get("eval/recall_weighted", 0.0)

            params_m = get_model_params(arch)

            data.append({
                "Model Name": arch,
                "Params (M)": params,
                "Batch Size": batch_size,
                "Top-1 Acc": acc1,
                "Top-3 Acc": acc3,
                "Top-5 Acc": acc5,
                "Weighted Precision": precision_weighted,
                "Weighted Recall": recall_weighted
            })

    if not data:
        print(f"Warning: No runs found with tag '{args.tag}' in project '{args.project}'.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by Params (M) for cleaner table output
    # Convert to numeric, force errors to NaN, then sort
    df["Params_Sort"] = pd.to_numeric(df["Params (M)"], errors='coerce')
    df = df.sort_values("Params_Sort")
    df = df.drop(columns=["Params_Sort"])
    
    print(f"Found {len(df)} run(s).")
    print(df)

    # --- Generate LaTeX Table ---
    print("\nGenerating LaTeX Table...")
    
    latex_str = "\\begin{table}[ht]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Final Zero-Shot Classification Results (Runs tagged: " + args.tag.replace("_", "\\_") + ")}\n"
    latex_str += "\\label{tab:tagged_results}\n"
    latex_str += "\\resizebox{\\textwidth}{!}{%\n"
    latex_str += "\\begin{tabular}{l c c c c c c c}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\textbf{Model} & \\textbf{Params (M)} & \\textbf{Batch Size} & \\textbf{Top-1} & \\textbf{Top-3} & \\textbf{Top-5} & \\textbf{W. Precision} & \\textbf{W. Recall} \\\\\n"
    latex_str += "\\midrule\n"

    for _, row in df.iterrows():
        line = f"{row['Model Name']} & "
        # Handle Params formatting if it's a number
        if isinstance(row['Params (M)'], (int, float)):
             line += f"{row['Params (M)']:.2f} & "
        else:
             line += f"{row['Params (M)']} & "
             
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
    filename = f"tagged_{args.tag}_table.tex"
    with open(filename, "w") as f:
        f.write(latex_str)
    print(f"\nSaved LaTeX table to {filename}")

if __name__ == "__main__":
    main()
