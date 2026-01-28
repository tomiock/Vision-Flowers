import pandas as pd
import argparse
import os
from scipy.stats import hypergeom, ttest_ind, ttest_rel

def load_data(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    df_classes = df.iloc[0:102].copy() # Filter for classes
    
    # Metric identification
    metric_col = None
    for col in ['f-score', 'f1-score', 'recall']:
        if col in df_classes.columns:
            metric_col = col
            break
            
    if metric_col is None:
        print(f"Error: No metric column found in {csv_path}")
        return None
    
    # Ensure numeric
    df_classes['Score'] = pd.to_numeric(df_classes[metric_col])
    
    # Assign generic class names if needed for set logic (indices 0-101 match class 1-102)
    # We use the index as the unique identifier for the set operations
    df_classes['Class_ID'] = df_classes.index 
    
    return df_classes, metric_col

def main():
    parser = argparse.ArgumentParser(description="Statistical Test for Worst Class Intersection")
    parser.add_argument("--fine_tuned_csv", type=str, default="table.csv")
    parser.add_argument("--baseline_csv", type=str, default="table_baseline.csv")
    args = parser.parse_args()

    # Load Data
    df_ft, _ = load_data(args.fine_tuned_csv, "Fine-Tuned")
    df_bl, _ = load_data(args.baseline_csv, "Baseline")

    if df_ft is None or df_bl is None:
        return

    # Get Worst 10 Class IDs
    worst_ft = set(df_ft.nsmallest(10, 'Score')['Class_ID'])
    worst_bl = set(df_bl.nsmallest(10, 'Score')['Class_ID'])

    # Parameters for Hypergeometric Test
    N = 102          # Total population size (number of flower classes)
    K = 10           # Size of the "success" group in population (Baseline worst 10)
    n = 10           # Sample size drawn (Fine-Tuned worst 10)
    k = len(worst_ft.intersection(worst_bl)) # Observed intersection size

    print(f"\n--- Intersection Analysis ---")
    print(f"Total Classes (N): {N}")
    print(f"Worst Classes per Model (n): {n}")
    print(f"Observed Intersection (k): {k}")

    # 1. Hypergeometric Test
    # Probability of getting intersection >= k by random chance
    # P(X >= k) = 1 - P(X < k) = 1 - cdf(k-1)
    p_value_hyper = 1 - hypergeom.cdf(k - 1, N, K, n)
    
    print(f"\n[Test 1] Hypergeometric Test (Overlap Significance)")
    print(f"Null Hypothesis: The worst-performing classes are randomly distributed.")
    print(f"P-value: {p_value_hyper:.4f}")
    if p_value_hyper < 0.05:
        print("-> Result: Statistically Significant (The overlap is larger than random chance)")
    else:
        print("-> Result: Not Significant (The overlap could be due to chance)")

    # 2. T-Test on Scores (Comparing the performance of these specific worst classes)
    # Are the scores of the worst 10 in Fine-Tuned statistically different from the worst 10 in Baseline?
    
    scores_ft_worst = df_ft.nsmallest(10, 'Score')['Score'].values
    scores_bl_worst = df_bl.nsmallest(10, 'Score')['Score'].values

    t_stat, p_value_t = ttest_ind(scores_ft_worst, scores_bl_worst)

    print(f"\n[Test 2] T-Test (Independent)")
    print(f"Comparing the means of the scores of the two worst-10 sets.")
    print(f"Fine-Tuned Worst-10 Mean: {scores_ft_worst.mean():.4f}")
    print(f"Baseline Worst-10 Mean:   {scores_bl_worst.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value_t:.4f}")
    
    if p_value_t < 0.05:
        print("-> Result: Statistically Significant Difference in Means")
    else:
        print("-> Result: No Significant Difference in Means")

if __name__ == "__main__":
    main()
