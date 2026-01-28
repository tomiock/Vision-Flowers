import pandas as pd
import argparse
import re

def parse_model_name(name):
    """
    Parses model name to identify base architecture and if it's extraction.
    Returns: (base_arch, is_extraction)
    """
    if "-Extraction" in name:
        base_arch = name.replace("-Extraction", "")
        return base_arch, True
    return name, False

def calculate_improvement(improved_val, baseline_val):
    """
    Calculates percentage increase of improved_val over baseline_val.
    Formula: ((New - Old) / Old) * 100
    """
    if baseline_val == 0: return 0.0
    return ((improved_val - baseline_val) / baseline_val) * 100

def main():
    # Hardcoded data from user provided table
    raw_data = [
        {"Model": "ViT-B/32-Extraction", "Params": 21.14, "Top1": 0.9814, "Top3": 0.9931, "Top5": 0.9980, "Runtime": "1 min"},
        {"Model": "ViT-L/14@336px-Extraction", "Params": 40.75, "Top1": 0.9990, "Top3": 1.0000, "Top5": 1.0000, "Runtime": "14 min"},
        {"Model": "ViT-B/32", "Params": 151.27, "Top1": 0.9588, "Top3": 0.9882, "Top5": 0.9931, "Runtime": "2 min"},
        {"Model": "ViT-L/14@336px", "Params": 427.94, "Top1": 0.9971, "Top3": 0.9990, "Top5": 1.0000, "Runtime": "24 min"}
    ]
    
    df = pd.DataFrame(raw_data)
    
    # Process data to pair models
    models = {}
    
    for _, row in df.iterrows():
        base, is_ext = parse_model_name(row['Model'])
        if base not in models:
            models[base] = {'ext': None, 'ft': None}
        
        type_key = 'ext' if is_ext else 'ft'
        models[base][type_key] = row

    print("\nGenerating Improvement Table (Extraction vs Fine-Tuning)...")
    
    latex_str = "\\begin{table}[ht]\n"
    latex_str += "\\centering\n"
    latex_str += "\\small\n"
    latex_str += "\\caption{Performance Gain: Feature Extraction vs. Full Fine-Tuning}\n"
    latex_str += "\\label{tab:improvement_results}\n"
    latex_str += "\\begin{adjustbox}{max width=\\linewidth}\n"
    latex_str += "\\begin{tabular}{l l c c c c c}\n"
    latex_str += "\\toprule\n"
    latex_str += "\\textbf{Architecture} & \\textbf{Method} & \\textbf{Params (M)} & \\textbf{Top-1} & \\textbf{Top-3} & \\textbf{Top-5} & \\textbf{Runtime} \\\\\n"
    latex_str += "\\midrule\n"

    for arch, data in models.items():
        ext = data['ext']
        ft = data['ft']
        
        if ext is None or ft is None:
            continue
            
        # Fine-Tuning Row (Baseline reference)
        latex_str += f"\\multirow{{2}}{{*}}{{{arch}}} & Fine-Tuned & {ft['Params']:.2f} & {ft['Top1']:.4f} & {ft['Top3']:.4f} & {ft['Top5']:.4f} & {ft['Runtime']} \\\\\n"
        
        # Calculate deltas (Extraction over FT)
        d_top1 = calculate_improvement(ext['Top1'], ft['Top1'])
        d_top3 = calculate_improvement(ext['Top3'], ft['Top3'])
        d_top5 = calculate_improvement(ext['Top5'], ft['Top5'])
        
        # Formatting delta strings
        def fmt_delta(val):
            # ForestGreen is standard for positive/good, requires xcolor with dvipsnames or define custom
            # Standard 'green' is sometimes too bright. Let's use 'green' or 'teal'.
            color = "green" if val >= 0 else "red"
            sign = "+" if val >= 0 else ""
            
            # If value is 0, show neutral
            if abs(val) < 0.001:
                return "\\scriptsize{(=)}"
                
            return f"\\scriptsize{{\\textcolor{{{color}}}{{({sign}{val:.2f}\\%)}}}}"

        # Feature Extraction Row with Deltas (Improvement)
        latex_str += f" & Extraction & {ext['Params']:.2f} & {ext['Top1']:.4f} {fmt_delta(d_top1)} & {ext['Top3']:.4f} {fmt_delta(d_top3)} & {ext['Top5']:.4f} {fmt_delta(d_top5)} & {ext['Runtime']} \\\\\n"
        
        # Add separator between groups if not last
        if arch != list(models.keys())[-1]:
            latex_str += "\\midrule\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{adjustbox}\n"
    latex_str += "\\end{table}"

    print(latex_str)
    
    with open("improvement_table.tex", "w") as f:
        f.write(latex_str)
    print("\nSaved to improvement_table.tex")

if __name__ == "__main__":
    main()
