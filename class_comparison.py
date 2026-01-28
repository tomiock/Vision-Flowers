import pandas as pd
import argparse
import os

def load_data(csv_path, label):
    """Loads classification report data from CSV and returns a DataFrame with class names and metric."""
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    
    # Filter rows 1-103 (indices 0 to 101, corresponding to classes 1-102)
    df_classes = df.iloc[0:102].copy()
    
    # Identify the metric column
    metric_col = None
    for col in ['f-score', 'f1-score', 'recall']:
        if col in df_classes.columns:
            metric_col = col
            break
            
    if metric_col is None:
        print(f"Error: No suitable metric column found in {csv_path}")
        return None
    
    return df_classes, metric_col

def get_class_names(data_root):
    import json
    cat_path = os.path.join(data_root, 'cat_to_name.json')
    if os.path.exists(cat_path):
        with open(cat_path, 'r') as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Create Table of Worst Performing Classes")
    parser.add_argument("--fine_tuned_csv", type=str, default="table.csv", help="Path to fine-tuned results CSV")
    parser.add_argument("--baseline_csv", type=str, default="table_baseline.csv", help="Path to baseline results CSV")
    parser.add_argument("--data_root", type=str, default="/data-net/storage2/datasets/OxfordF", help="Path to dataset root")
    args = parser.parse_args()

    # Load class names
    cat_to_name = get_class_names(args.data_root)
    
    # Load Dataframes
    df_ft, metric_ft = load_data(args.fine_tuned_csv, "Fine-Tuned")
    df_bl, metric_bl = load_data(args.baseline_csv, "Baseline")

    if df_ft is None or df_bl is None:
        return

    # Add Class Names
    class_names = []
    for idx in range(len(df_ft)):
        class_id = str(idx + 1)
        name = cat_to_name[class_id] if cat_to_name else f"Class {class_id}"
        class_names.append(name)

    df_ft['Class'] = class_names
    df_bl['Class'] = class_names
    
    # Extract Score columns
    df_ft['Score'] = pd.to_numeric(df_ft[metric_ft])
    df_bl['Score'] = pd.to_numeric(df_bl[metric_bl])

    # Get Worst 20 for each (Matches the table you showed)
    N_WORST = 20
    worst_ft = df_ft.nsmallest(N_WORST, 'Score')
    worst_bl = df_bl.nsmallest(N_WORST, 'Score')

    # Calculate Intersection
    ft_worst_set = set(worst_ft['Class'])
    bl_worst_set = set(worst_bl['Class'])
    intersection_classes = ft_worst_set.intersection(bl_worst_set)
    
    print(f"\nComparing top {N_WORST} worst classes.")
    print(f"Number of classes in intersection: {len(intersection_classes)}")
    print(f"Intersection: {intersection_classes}")

    # Format for LaTeX
    print("\nGenerating LaTeX Table...")
    
    # Header with package requirement note
    latex_str = "% Requires \\usepackage{xcolor} in preamble\n"
    latex_str += "\\begin{table}[ht]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Comparison of the " + str(N_WORST) + " Worst Performing Classes (Baseline vs. Fine-Tuned). \\textbf{Bold/Red} indicates classes present in both lists.}\n"
    latex_str += "\\label{tab:worst_classes}\n"
    latex_str += "\\resizebox{\\textwidth}{!}{%\n"
    latex_str += "\\begin{tabular}{c l c | l c}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\textbf{Rank} & \\textbf{Baseline Class} & \\textbf{F1} & \\textbf{Fine-Tuned Class} & \\textbf{F1} \\\\\n"
    latex_str += "\\midrule\n"

    worst_bl = worst_bl.reset_index(drop=True)
    worst_ft = worst_ft.reset_index(drop=True)

    for i in range(N_WORST):
        rank = i + 1
        bl_name = worst_bl.iloc[i]['Class']
        bl_score = worst_bl.iloc[i]['Score']
        ft_name = worst_ft.iloc[i]['Class']
        ft_score = worst_ft.iloc[i]['Score']
        
        # Apply formatting if in intersection
        # Using \textcolor{red}{\textbf{Name}} for visibility
        
        if bl_name in intersection_classes:
            bl_name_fmt = f"\\textcolor{{red}}{{\\textbf{{{bl_name}}}}}"
        else:
            bl_name_fmt = bl_name
            
        if ft_name in intersection_classes:
            ft_name_fmt = f"\\textcolor{{red}}{{\\textbf{{{ft_name}}}}}"
        else:
            ft_name_fmt = ft_name
        
        line = f"{rank} & {bl_name_fmt} & {bl_score:.2f} & {ft_name_fmt} & {ft_score:.2f} \\\\"
        latex_str += line + "\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}}\n"
    latex_str += "\\end{table}"

    print(latex_str)
    
    with open("worst_classes_table.tex", "w") as f:
        f.write(latex_str)
    print("\nSaved to worst_classes_table.tex")

if __name__ == "__main__":
    main()
