import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_experiments(runs_dir='runs/detect'):
    """
    Finds all experiment folders, extracts their best performance,
    and generates a comparison table and plot
    """
    
    # key = display name, value = folder name in runs/detect/
    experiments = {
        'Baseline (YOLOv8n)': 'yolov8n_baseline_fish4knowledge_sdk',
        '+ Focal Loss': 'yolov8n_focal_loss',
        '+ CBAM': 'yolov8n_cbam',
        '+ ECA': 'yolov8n_eca',
        '+ BiFPN': 'yolov8n_bifpn'
        # Add any other experiments you ran
    }
    
    results_list = []
    
    print("--- Analyzing Experiment Results ---")
    
    # loop through each experiment folder and extract the best metrics
    for display_name, folder_name in experiments.items():
        exp_path = Path(runs_dir) / folder_name
        results_csv_path = exp_path / 'results.csv'
        
        if not results_csv_path.exists():
            print(f"⚠️  Warning: Could not find results.csv for '{display_name}'. Skipping.")
            continue
        
        df = pd.read_csv(results_csv_path)
  
        df.columns = df.columns.str.strip()
        
        # find the best epoch based on the highest mAP50-95 score
        best_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
        
        # Extract the key metrics from that best epoch
        result = {
            'Experiment': display_name,
            'mAP50-95': best_epoch['metrics/mAP50-95(B)'],
            'mAP50': best_epoch['metrics/mAP50(B)'],
            'Precision': best_epoch['metrics/precision(B)'],
            'Recall': best_epoch['metrics/recall(B)']
        }
        results_list.append(result)
        print(f"✅ Processed '{display_name}' - Best mAP50-95: {result['mAP50-95']:.4f}")

    if not results_list:
        print("❌ No experiment results found. Please check your folder names.")
        return

    # --- 3. Create and display the final comparison table ---
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='mAP50-95', ascending=False)
    results_df = results_df.set_index('Experiment')
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(float_format="%.4f"))
    print("="*80)
    
    # --- 4. Generate a bar chart for your report ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    results_df['mAP50-95'].sort_values().plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title('Ablation Study: Impact of Custom Modules on mAP50-95', fontsize=16)
    ax.set_xlabel('mAP@.50:.95', fontsize=12)
    ax.set_ylabel('Model Configuration', fontsize=12)
    
    # Add value labels to the bars
    for i, v in enumerate(results_df['mAP50-95'].sort_values()):
        ax.text(v + 0.001, i, f"{v:.4f}", color='black', va='center')
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png')
    print("\n Bar chart saved to 'ablation_study_results.png'")
    plt.show()

def main():
    analyze_experiments()

if __name__ == '__main__':
    main()