import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(csv_path: str, out_dir: str):
    """Generates standard benchmark plots from the aggregated metric CSV."""
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}. Cannot visualize.")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        print("Dataframe empty. Skipping visualization.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup aesthetic defaults
    sns.set_theme(style="whitegrid", palette="Set2")
    
    # Average across seeds (if multiple seeds exist for the case_id + model)
    # This prevents statistical inflation for multi-seed models compared to MedSAM (1 seed)
    df_avg_seed = df.groupby(["model", "case_id"], as_index=False).mean(numeric_only=True)
    
    models_present = df_avg_seed['model'].unique()
    
    # 1. Dice Boxplot
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=df_avg_seed, x="model", y="dice", hue="model", dodge=False)
    plt.title("Dice Similarity Coefficient by Model (Averaged across seeds)")
    plt.ylabel("Dice Score")
    plt.xlabel("")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_boxplot.png"), dpi=300)
    plt.close()
    
    # 2. IoU Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_avg_seed, x="model", y="iou", hue="model", dodge=False)
    plt.title("Intersection over Union (IoU) by Model")
    plt.ylabel("IoU Score")
    plt.xlabel("")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_boxplot.png"), dpi=300)
    plt.close()
    
    # 3. HD95 Boxplot (Lower is better, filter out NaNs or extremely large errors for legibility)
    plt.figure(figsize=(8, 6))
    hd95_df = df_avg_seed.dropna(subset=['hd95'])
    # Cap HD95 at 99th percentile for visualization to avoid single huge outliers destroying the plot
    if not hd95_df.empty:
        cap = hd95_df['hd95'].quantile(0.99)
        capped_df = hd95_df.copy()
        capped_df['hd95'] = capped_df['hd95'].clip(upper=cap)
        
        sns.boxplot(data=capped_df, x="model", y="hd95", hue="model", dodge=False)
        plt.title("Hausdorff Distance 95% (Lower is better)")
        plt.ylabel("Distance (pixels/mm)")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hd95_boxplot.png"), dpi=300)
    plt.close()
    
    print(f"Generated visualization plots in {out_dir}")

if __name__ == "__main__":
    create_visualizations("reports/benchmark_results.csv", "reports/figures")
