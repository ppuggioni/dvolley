import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from models import LogisticRegressionModel, EmpiricalModel, SimpleEmpiricalModel, GlobalMeanModel
from sklearn.metrics import log_loss, brier_score_loss

def plot_calibration_quantile(y_true, y_pred, model_name, output_dir, min_samples_per_bin=400):
    """Plot calibration with smart quantile binning (approx equal sizes >= min_samples)."""
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Sort by prediction
    sorted_indices = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    n_samples = len(y_pred)
    
    # Calculate number of bins
    n_bins = n_samples // min_samples_per_bin
    if n_bins < 1: n_bins = 1
    
    # Calculate base size and remainder
    base_size = n_samples // n_bins
    remainder = n_samples % n_bins
    
    prob_true = []
    prob_pred = []
    bin_counts = []
    
    current_idx = 0
    for i in range(n_bins):
        # Distribute remainder across first 'remainder' bins
        extra = 1 if i < remainder else 0
        bin_size = base_size + extra
        
        end_idx = current_idx + bin_size
        
        bin_y_true = y_true_sorted[current_idx:end_idx]
        bin_y_pred = y_pred_sorted[current_idx:end_idx]
        
        prob_true.append(np.mean(bin_y_true))
        prob_pred.append(np.mean(bin_y_pred))
        bin_counts.append(len(bin_y_true))
        
        current_idx = end_idx
    
    # Determine plot range to show ALL points
    all_values = prob_true + prob_pred
    if not all_values:
        plot_min, plot_max = 0, 1
    else:
        plot_min = min(all_values) - 0.02
        plot_max = max(all_values) + 0.02
        plot_min = max(0, plot_min)
        plot_max = min(1, plot_max)
    
    # Plot
    plt.figure(figsize=(10, 10))
    
    # Perfect calibration line
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', 
             linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Calibration curve
    if prob_pred:
        # Calculate 95% CI for each bin
        p_hat = np.array(prob_true)
        ns = np.array(bin_counts)
        p_hat_safe = np.clip(p_hat, 0.01, 0.99)
        se = np.sqrt(p_hat_safe * (1 - p_hat_safe) / ns)
        yerr = 1.96 * se
        
        # Plot error bars
        plt.errorbar(prob_pred, prob_true, yerr=yerr, fmt='none', 
                     ecolor='gray', alpha=0.5, capsize=3, label='95% CI')
        
        # Plot points
        plt.scatter(prob_pred, prob_true, s=150, alpha=0.7, 
                   label=f'{model_name} ({n_bins} bins, ~{base_size}/bin)')
        plt.plot(prob_pred, prob_true, '-', alpha=0.4, linewidth=2, color='C0')
        
        # Add sample counts
        for i, (x, y, c) in enumerate(zip(prob_pred, prob_true, bin_counts)):
            plt.text(x, y, f'n={c}', fontsize=8, alpha=0.7, 
                    ha='center', va='bottom')
    
    plt.xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    plt.ylabel('Fraction of Positives (Actual)', fontsize=13, fontweight='bold')
    plt.title(f'Calibration: {model_name}\nLOOCV - All Matches (Smart Binning)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    filename = f"calibration_{model_name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved calibration plot: {filename} ({n_bins} bins, sizes: {min(bin_counts)}-{max(bin_counts)})")

def main():
    # Load Data
    csv_path = "clean_data/clean_data.csv"
    df = pd.read_csv(csv_path, encoding="cp1252")
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Prepare data
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date")
    df["match_id"] = (df["match_date"].astype(str) + "_" + 
                      df["team_id_h"].astype(str) + "_" + 
                      df["team_id_a"].astype(str))
    
    unique_matches = df["match_id"].unique()
    print(f"Unique matches: {len(unique_matches)}\n")
    
    # Define Models
    models = [
        GlobalMeanModel(),
        SimpleEmpiricalModel(),
        EmpiricalModel(),
        LogisticRegressionModel(alpha=0.1),
        LogisticRegressionModel(alpha=0.01),
        LogisticRegressionModel(alpha=0.001)
    ]
    
    # Run LOOCV
    print("Running Leave-One-Match-Out CV...")
    results = {model.name: {"y_true": [], "y_pred": []} for model in models}
    
    for i, match_id in enumerate(unique_matches):
        if (i + 1) % 5 == 0:
            print(f"  Processing match {i+1}/{len(unique_matches)}...")
        
        train_mask = df["match_id"] != match_id
        test_mask = df["match_id"] == match_id
        
        df_train = df[train_mask]
        df_test = df[test_mask]
        
        # Calculate y_test
        serve_is_home = (df_test["serve_team"] == "h")
        y_test = np.where(
            (serve_is_home & (df_test["point_won_team"] == "h")) |
            (~serve_is_home & (df_test["point_won_team"] == "a")),
            1, 0
        )
        
        # Train and predict
        for model in models:
            model.fit(df_train)
            y_prob = model.predict_proba(df_test)
            
            results[model.name]["y_true"].extend(y_test)
            results[model.name]["y_pred"].extend(y_prob)
    
    # Evaluate and plot
    output_dir = "loocv_calibration_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*85)
    print("LOOCV Results - All Matches")
    print("="*85)
    print(f"{'Model':<20} {'Log Loss':>12} {'Brier Score':>12} {'Accuracy':>12} {'Samples':>10}")
    print("-"*85)
    
    metrics = []
    
    for model_name in results:
        y_true = np.array(results[model_name]["y_true"])
        y_pred = np.array(results[model_name]["y_pred"])
        
        ll = log_loss(y_true, y_pred, labels=[0, 1])
        bs = brier_score_loss(y_true, y_pred)
        acc = np.mean((y_pred > 0.5) == y_true)
        
        metrics.append({
            "Model": model_name,
            "Log Loss": ll,
            "Brier Score": bs,
            "Accuracy": acc,
            "Samples": len(y_true)
        })
        
    # Sort by Log Loss
    metrics.sort(key=lambda x: x["Log Loss"])
    
    # Save to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "metrics_ranking.csv"), index=False)
    print(f"Metrics ranking saved to {os.path.join(output_dir, 'metrics_ranking.csv')}")
    
    for m in metrics:
        print(f"{m['Model']:<20} {m['Log Loss']:12.4f} {m['Brier Score']:12.4f} {m['Accuracy']*100:11.2f}% {m['Samples']:10d}")
        
        # Plot calibration
        # Re-find the y_true/y_pred for plotting
        y_true = np.array(results[m['Model']]["y_true"])
        y_pred = np.array(results[m['Model']]["y_pred"])
        plot_calibration_quantile(y_true, y_pred, m['Model'], output_dir, min_samples_per_bin=500)
    
    print("="*85)
    print(f"\nCalibration plots saved to '{output_dir}/'")

if __name__ == "__main__":
    main()
