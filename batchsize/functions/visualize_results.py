
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

def visualize_batch_size_results(project_root, model_name='resnet50'):
    """
    Visualizes best accuracy from logs for different batch sizes.
    
    Args:
        project_root (str): The root directory of the project containing the 'logs' folder.
        model_name (str): The name of the model ('resnet50', etc.) to filter logs.
    """
    
    # Visualization of Best Accuracy from Logs (Fixed Logic)
    log_dir = os.path.join(project_root, 'logs')
    log_files = [f for f in os.listdir(log_dir) if f.startswith(f'{model_name}_bs') and f.endswith('.log')]

    best_results = []
    all_results = []

    for filename in log_files:
        bs_match = re.search(r'bs(\d+)', filename)
        if not bs_match: continue
        bs = int(bs_match.group(1))
        
        with open(os.path.join(log_dir, filename), 'r') as f:
            content = f.read()
        
        # Split by "Training started." and take the last run
        runs = content.split("Training started.")
        if len(runs) < 2: continue
        last_run = runs[-1]
        
        # Extract Accuracy for all epochs
        epochs_raw = re.findall(r'Epoch \[(\d+)/\d+\]', last_run)
        train_accs_raw = re.findall(r'Train Acc: ([\d.]+)', last_run)
        val_accs_raw = re.findall(r'Val\s+Acc: ([\d.]+)', last_run)
        
        if epochs_raw and train_accs_raw and val_accs_raw:
            # zip together and iterate, safe way to handle potentially mismatched lengths if logs are corrupted
            # though regex lengths should match if log format is consistent.
            # Convert to float/int
            try:
                epochs = [int(x) for x in epochs_raw]
                train_accs = [float(x) for x in train_accs_raw]
                val_accs = [float(x) for x in val_accs_raw]
            except ValueError:
                continue

            min_len = min(len(epochs), len(train_accs), len(val_accs))
            
            run_data = []
            best_val_acc_in_run = -1.0
            
            # First pass: collect data and find best validation accuracy
            temp_run_data = []
            for i in range(min_len):
                ep = epochs[i]
                t_acc = train_accs[i]
                v_acc = val_accs[i]
                diff = round(t_acc - v_acc, 4)
                
                if v_acc > best_val_acc_in_run:
                    best_val_acc_in_run = v_acc
                
                temp_run_data.append({
                    'Batch Size': bs,
                    'Epoch': ep,
                    'Train Acc': t_acc,
                    'Val Acc': v_acc,
                    'Diff': diff
                })
            
            # Second pass: mark best and add to lists
            for item in temp_run_data:
                is_best = (item['Val Acc'] == best_val_acc_in_run)
                # If multiple epochs have same best val acc, we mark all of them or just first? 
                # Let's say all for now or user code logic was just checking max.
                # User code logic: "if v_acc > best_val_acc: best - else not best".
                # But it was iterating. So only the *last* highest one would be the 'best' if stricly >.
                # Let's stick to user logic: Find max, then mark.
                
                # Re-evaluating user logic:
                # User logic: `if v_acc > best_val_acc: best_val_acc = v_acc; best_epoch_idx = ep`
                # Then loop again run_data matching epoch.
                # If multiple epochs have equal max score, user logic takes the FIRST one (strict greater).
                
                item['Is Best'] = (item['Val Acc'] == best_val_acc_in_run)
                all_results.append(item)
                
                if item['Is Best']:
                    # Check if we already have an entry for this BS (in case of duplicate bests, pick first or all?)
                    # User code logic adds to `best_results` inside the loop. 
                    # If multiple epochs are "best" (same value), they all get added.
                    # We should probably deduplicate or take the one with better train acc/lower diff?
                    # For now, let's keep it simple and just take the first one or distinct.
                    # Actually, let's filter `best_results` to unique Batch Size later if needed.
                     best_results.append({
                        'Batch Size': bs,
                        'Best Epoch': item['Epoch'],
                        'Train Acc (Best)': item['Train Acc'],
                        'Val Acc (Best)': item['Val Acc'],
                        'Diff (Best)': item['Diff']
                    })

    # Sort and deduplicate best_results by Batch Size (taking the one with highest Val Acc, then maybe latest epoch?)
    # Simply sorting by Batch Size is enough for plotting.
    if best_results:
        # If multiple 'best' epochs for same BS, dropped duplicates or keep? 
        # Visualization expects one point per BS usually.
        df_best = pd.DataFrame(best_results).sort_values('Batch Size').drop_duplicates(subset=['Batch Size'], keep='first')
        
        plt.figure(figsize=(12, 7))
        x_positions = range(len(df_best))
        bs_labels = df_best['Batch Size'].astype(str).tolist()
        
        plt.plot(x_positions, df_best['Train Acc (Best)'], 
                 marker='o', linestyle='-', color='blue', label='Train Accuracy (Best Accuracy Epoch)')
        plt.plot(x_positions, df_best['Val Acc (Best)'], 
                 marker='s', linestyle='--', color='red', label='Val Accuracy (Best Accuracy Epoch)')
        
        # Annotate differences
        for i, row in df_best.iterrows():
            # i is index from dataframe which might be non-sequential after sort/filter
            # We need the enumeration index for x_axis
            # Let's convert to list dicts for safe iteration with zip
            pass # See next block for correct iteration
            
        # Correct iteration for plotting
        for idx in range(len(df_best)):
            row = df_best.iloc[idx]
            x = idx
            y_train = row['Train Acc (Best)']
            y_val = row['Val Acc (Best)']
            diff = row['Diff (Best)']
            
            # Draw a line between train and val
            plt.vlines(x, y_val, y_train, color='gray', linestyle=':', alpha=0.5)
            
            # Annotate text (Delta)
            text_y = (y_train + y_val) / 2
            plt.text(x, text_y, f'Δ:{diff:.4f}', 
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                     fontsize=9, fontweight='bold', color='darkgreen')

        plt.title(f'Best Accuracy vs Batch Size for {model_name}', fontsize=16)
        plt.xlabel('Batch Size', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(x_positions, bs_labels)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 2. Detailed Table
        df_all = pd.DataFrame(all_results).sort_values(['Batch Size', 'Epoch'])
        print("\n--- Detailed Results (Corrected Diff & Accuracy-Based Best Highlights) ---")
        
        # Check if running in Jupyter/IPython for styling
        try:
            # Use pandas Styler if available
            styler = df_all.style.format({'Train Acc': '{:.4f}', 'Val Acc': '{:.4f}', 'Diff': '{:.4f}'}) \
                    .map(lambda x: 'font-weight: bold; color: green' if x else '', subset=['Is Best']) \
                    .hide(axis='index')
            display(styler)
        except (NameError, ImportError):
            # Fallback for terminal
            print(df_all.to_string(index=False))
            
    else:
        print("No valid training summaries found in logs.")

if __name__ == "__main__":
    # Example usage if run directly
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Batch Size Experiment Results")
    parser.add_argument("--project_root", type=str, default=os.getcwd(), help="Project root directory")
    parser.add_argument("--model", type=str, default='resnet50', help="Model name")
    
    args = parser.parse_args()
    
    try:
        visualize_batch_size_results(args.project_root, args.model)
    except Exception as e:
        print(f"Error visualizing results: {e}")
