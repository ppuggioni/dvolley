import pandas as pd
from models import LogisticRegressionModel, EmpiricalModel, SimpleEmpiricalModel, GlobalMeanModel
from backtest_engine import BackTestEngine
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import log_loss, brier_score_loss

def main():
    # Load Data
    csv_path = "clean_data/clean_data.csv"
    df = pd.read_csv(csv_path, encoding="cp1252")
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Filter for Mantova matches only (for evaluation)
    mantova_mask = (df["team_h"] == "Gabbiano FarmaMed Mantova") | (df["team_a"] == "Gabbiano FarmaMed Mantova")
    df_mantova = df[mantova_mask].copy()
    
    print(f"\nGabbiano FarmaMed Mantova matches: {len(df_mantova)} rows")
    print(f"Other matches: {len(df[~mantova_mask])} rows")
    
    # Define Models
    models = [
        GlobalMeanModel(),
        SimpleEmpiricalModel(),
        EmpiricalModel(),
        LogisticRegressionModel(alpha=0.1),
        LogisticRegressionModel(alpha=0.01),
        LogisticRegressionModel(alpha=0.001)
    ]
    
    # Manual LOOCV for Mantova matches
    df_mantova["match_date"] = pd.to_datetime(df_mantova["match_date"], format='mixed', dayfirst=True)
    df_mantova = df_mantova.sort_values("match_date")
    df_mantova["match_id"] = (df_mantova["match_date"].astype(str) + "_" + 
                               df_mantova["team_id_h"].astype(str) + "_" + 
                               df_mantova["team_id_a"].astype(str))
    
    df["match_date"] = pd.to_datetime(df["match_date"], format='mixed', dayfirst=True)
    df["match_id"] = (df["match_date"].astype(str) + "_" + 
                      df["team_id_h"].astype(str) + "_" + 
                      df["team_id_a"].astype(str))
    
    mantova_matches = df_mantova["match_id"].unique()
    print(f"Unique Mantova matches: {len(mantova_matches)}")
    
    # Run LOOCV
    results = {model.name: {"y_true": [], "y_pred": []} for model in models}
    
    for i, match_id in enumerate(mantova_matches):
        print(f"Processing Mantova match {i+1}/{len(mantova_matches)}...")
        
        # Train on all OTHER matches (including non-Mantova)
        train_mask = df["match_id"] != match_id
        test_mask = df_mantova["match_id"] == match_id
        
        df_train = df[train_mask]
        df_test = df_mantova[test_mask]
        
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
    
    # Calculate metrics
    output_dir = "mantova_loocv_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*60)
    print("LOOCV Results for Gabbiano FarmaMed Mantova Matches")
    print("="*60)
    print(f"{'Model':<20} {'Log Loss':>12} {'Accuracy':>12} {'Samples':>10}")
    print("-"*60)
    
    for model_name in results:
        y_true = np.array(results[model_name]["y_true"])
        y_pred = np.array(results[model_name]["y_pred"])
        
        ll = log_loss(y_true, y_pred, labels=[0, 1])
        acc = np.mean((y_pred > 0.5) == y_true)
        
        print(f"{model_name:<20} {ll:12.4f} {acc*100:11.2f}% {len(y_true):10d}")
        
        # Plot calibration
        plot_calibration(y_true, y_pred, model_name, output_dir, target_samples_per_bin=30)
    
    print("="*60)
    print(f"\nCalibration plots saved to '{output_dir}/'")

def plot_calibration(y_true, y_pred, model_name, output_dir, target_samples_per_bin=30):
    """Plot calibration with adaptive binning targeting N samples per bin."""
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Determine prediction range
    pred_min = np.min(y_pred)
    pred_max = np.max(y_pred)
    
    # Add small margin
    margin = (pred_max - pred_min) * 0.05
    plot_min = max(0, pred_min - margin)
    plot_max = min(1, pred_max + margin)
    
    # Calculate number of bins based on target samples
    n_samples = len(y_pred)
    n_bins = max(5, min(20, n_samples // target_samples_per_bin))
    
    # Create bins across the prediction range
    bins = np.linspace(plot_min, plot_max, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins)
    
    prob_true = []
    prob_pred = []
    bin_counts = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            prob_true.append(np.mean(y_true[mask]))
            prob_pred.append(np.mean(y_pred[mask]))
            bin_counts.append(np.sum(mask))
    
    # Plot
    plt.figure(figsize=(8, 8))
    
    # Perfect calibration line
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', 
             linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Calibration curve with bubble sizes
    if prob_pred:
        sizes = np.array(bin_counts) / max(bin_counts) * 300 + 50
        plt.scatter(prob_pred, prob_true, s=sizes, alpha=0.6, 
                   label=f'{model_name} (bins={len(prob_pred)})')
        plt.plot(prob_pred, prob_true, '-', alpha=0.3, linewidth=2)
        
        # Add bin counts as text
        for i, (x, y, c) in enumerate(zip(prob_pred, prob_true, bin_counts)):
            if i % 2 == 0:  # Label every other bin to avoid clutter
                plt.text(x, y, f'n={c}', fontsize=8, alpha=0.7, 
                        ha='center', va='bottom')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives (Actual)', fontsize=12)
    plt.title(f'Calibration: {model_name}\nMantova Matches (LOOCV)', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Set limits to relevant range
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    
    # Make it square
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    filename = f"calibration_{model_name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
