import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from analysis_regr import VolleyballBreakpointSideoutRegModelNoHome
import os

def main():
    csv_path = "clean_data/clean_data.csv"
    alphas = [0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    
    # Load Data
    try:
        df = pd.read_csv(csv_path, encoding="cp1252", parse_dates=["match_date"], dayfirst=True)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    # Identify Matches
    # Group by date and teams to get unique matches
    # We can create a 'match_id' column
    df['match_id'] = df['match_date'].astype(str) + "_" + df['team_id_h'].astype(str) + "_" + df['team_id_a'].astype(str)
    match_ids = df['match_id'].unique()
    
    print(f"Found {len(match_ids)} matches.")
    print("-" * 60)
    
    results = {}
    
    for alpha in alphas:
        print(f"\nEvaluating Alpha = {alpha}")
        
        total_log_loss = 0
        all_y_true = []
        all_y_pred = []
        
        # Leave-One-Match-Out CV
        for mid in match_ids:
            # Split
            test_mask = df['match_id'] == mid
            train_mask = ~test_mask
            
            df_train = df[train_mask].copy()
            df_test = df[test_mask].copy()
            
            # Fit
            model = VolleyballBreakpointSideoutRegModelNoHome(alpha=alpha)
            # We need to load data into the model to initialize it
            # But the model class expects a CSV path usually.
            # However, we can bypass load_data and set attributes directly if we are careful,
            # OR we can save temp CSVs. Saving temp CSVs is safer/easier given the class structure.
            
            # Actually, the model class is designed to load from CSV.
            # Let's modify the class slightly? No, let's just save temp files.
            # It's a bit slow but robust.
            
            df_train.to_csv("temp_train.csv", index=False)
            model.load_data("temp_train.csv")
            model.fit()
            
            # Predict
            # We need y_test for scoring
            # The model's load_data calculates y. We can replicate that logic or just use the column if we trust it.
            # Let's replicate the target logic to be safe:
            serve_is_home = (df_test["serve_team"] == "h")
            point_won_team = df_test["point_won_team"]
            y_test = np.where(
                (serve_is_home & (point_won_team == "h")) | (~serve_is_home & (point_won_team == "a")),
                1, 0
            )
            
            try:
                y_prob = model.predict_proba(df_test)
                
                # Debug: print first few to verify
                if mid == match_ids[0] and alpha == alphas[0]:
                    print(f"\n  DEBUG first 10 predictions:")
                    for i in range(min(10, len(y_test))):
                        print(f"    Pred: {y_prob[i]:.3f}, Actual: {y_test[i]}")
                
                # Accumulate
                loss = log_loss(y_test, y_prob, labels=[0, 1])
                # Weighted by number of samples in this match
                total_log_loss += loss * len(y_test)
                
                all_y_true.extend(y_test)
                all_y_pred.extend(y_prob)
                
            except Exception as e:
                print(f"Error predicting for match {mid}: {e}")
        
        # Calculate Average Log Loss
        avg_log_loss = total_log_loss / len(all_y_true)
        results[alpha] = avg_log_loss
        
        # Debug info
        print(f"  Avg Log Loss: {avg_log_loss:.5f}")
        print(f"  Prediction Range: [{np.min(all_y_pred):.4f}, {np.max(all_y_pred):.4f}]")
        print(f"  Actual Mean: {np.mean(all_y_true):.4f}")
        
        # Calibration Plot
        plot_calibration(all_y_true, all_y_pred, alpha)

    # Clean up
    if os.path.exists("temp_train.csv"):
        os.remove("temp_train.csv")

    print("\n" + "=" * 60)
    print("Optimization Results:")
    best_alpha = min(results, key=results.get)
    for alpha, loss in results.items():
        mark = "*" if alpha == best_alpha else ""
        print(f"Alpha {alpha}: {loss:.5f} {mark}")
        
    print(f"\nBest Alpha: {best_alpha}")

def plot_calibration(y_true, y_pred, alpha):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Adaptive binning based on actual prediction range
    pred_min = np.min(y_pred)
    pred_max = np.max(y_pred)
    
    # Add 5% margin on each side
    margin = 0.05
    bin_min = max(0, pred_min - margin)
    bin_max = min(1, pred_max + margin)
    
    # Use 1% bins (100 bins in 0-1 range, but only in our range)
    n_bins = int((bin_max - bin_min) * 100) + 1
    bins = np.linspace(bin_min, bin_max, n_bins)
    bin_indices = np.digitize(y_pred, bins)
    
    prob_true = []
    prob_pred = []
    bin_counts = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            prob_true.append(np.mean(y_true[mask]))
            prob_pred.append(np.mean(y_pred[mask]))
            bin_counts.append(np.sum(mask))
    
    plt.figure(figsize=(8, 8))
    
    # Plot diagonal
    diag_range = [bin_min, bin_max]
    plt.plot(diag_range, diag_range, "k--", label="Perfectly Calibrated", linewidth=2)
    
    # Plot calibration with size proportional to bin count
    sizes = np.array(bin_counts) / np.max(bin_counts) * 200 + 20
    plt.scatter(prob_pred, prob_true, s=sizes, alpha=0.6, label=f"Alpha={alpha}")
    plt.plot(prob_pred, prob_true, "-", alpha=0.3)
    
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Actual)", fontsize=12)
    plt.title(f"Calibration Plot (Alpha={alpha})\nBubble size = # samples in bin", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to the relevant range
    plt.xlim(bin_min, bin_max)
    plt.ylim(bin_min, bin_max)
    
    plt.savefig(f"calibration_alpha_{alpha}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print calibration table
    print(f"\n  Calibration Data for Alpha={alpha}:")
    print(f"  {'Pred%':>8s} {'Actual%':>8s} {'Count':>8s}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}")
    for pred, actual, count in zip(prob_pred, prob_true, bin_counts):
        print(f"  {pred*100:8.2f} {actual*100:8.2f} {count:8d}")
    
    print(f"  Calibration plot saved with {len(prob_pred)} bins")

if __name__ == "__main__":
    main()
