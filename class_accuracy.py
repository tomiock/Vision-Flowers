import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def load_data(csv_path, label):
    """Loads classification report data from CSV and returns a DataFrame."""
    print(f"Loading data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    
    # Filter rows 1-103 (indices 0 to 101, corresponding to classes 1-102)
    # Assumes standard export format where first 102 rows are classes
    df_classes = df.iloc[0:102].copy()
    
    # Identify the metric column (usually 'f-score' or 'f1-score' or 'recall')
    # Prioritize f1-score, fallback to others
    metric_col = None
    for col in ['f-score', 'f1-score', 'recall', 'precision']:
        if col in df_classes.columns:
            metric_col = col
            break
            
    if metric_col is None:
        print(f"Error: No suitable metric column found in {csv_path}")
        return None
        
    df_classes['Metric'] = pd.to_numeric(df_classes[metric_col])
    df_classes['Run'] = label
    
    print(f"Loaded {len(df_classes)} classes for '{label}' using metric '{metric_col}'")
    return df_classes[['Metric', 'Run']]

def main():
    parser = argparse.ArgumentParser(description="Compare Accuracy Distributions from Two CSVs")
    parser.add_argument("--fine_tuned_csv", type=str, default="table.csv", help="Path to fine-tuned results CSV")
    parser.add_argument("--baseline_csv", type=str, default="table_baseline.csv", help="Path to baseline results CSV")
    parser.add_argument("--output_file", type=str, default="accuracy_distribution_comparison.png", help="Output filename")
    args = parser.parse_args()

    # Load data
    df_ft = load_data(args.fine_tuned_csv, "Fine-Tuned")
    df_bl = load_data(args.baseline_csv, "Baseline (Zero-Shot)")
    
    if df_ft is None or df_bl is None:
        print("Exiting due to data loading errors.")
        return

    # Combine dataframes
    df_combined = pd.concat([df_ft, df_bl], ignore_index=True)

    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Histogram with KDE
    # 'layer="step"' makes it easier to see overlap, 'element="step"' creates outline-style bars
    # 'common_norm=False' calculates density independently for each group
    sns.histplot(
        data=df_combined,
        x="Metric",
        hue="Run",
        bins=50,
        kde=True,
        element="step",
        stat="count",
        common_norm=False,
        palette={"Fine-Tuned": "tab:blue", "Baseline (Zero-Shot)": "tab:orange"}
    )

    # Add mean lines
    mean_ft = df_ft['Metric'].mean()
    mean_bl = df_bl['Metric'].mean()
    
    plt.axvline(mean_ft, color='tab:blue', linestyle='--', linewidth=2, label=f'Fine-Tuned Mean: {mean_ft:.2f}')
    plt.axvline(mean_bl, color='tab:orange', linestyle='--', linewidth=2, label=f'Baseline Mean: {mean_bl:.2f}')

    plt.title("Distribution of Class Accuracies: Fine-Tuned vs Baseline (bins=50)", fontsize=15)
    plt.xlabel("Class Accuracy (F1-Score)", fontsize=12)
    plt.ylabel("Number of Classes", fontsize=12)
    plt.xlim(0, 1.0)
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Comparison plot saved to {args.output_file}")
    plt.show()

if __name__ == "__main__":
    main()
