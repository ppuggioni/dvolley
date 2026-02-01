import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, brier_score_loss
import io
import base64

def calculate_metrics(y_true, y_pred):
    """Calculate Log Loss, Brier Score, and Accuracy."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Clip predictions for log loss stability
    y_pred_safe = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    ll = log_loss(y_true, y_pred_safe, labels=[0, 1])
    bs = brier_score_loss(y_true, y_pred)
    acc = np.mean((y_pred > 0.5) == y_true)
    
    return {
        "Log Loss": ll,
        "Brier Score": bs,
        "Accuracy": acc,
        "Samples": len(y_true)
    }

def run_loocv_backtest(model, df):
    """Run Leave-One-Match-Out Cross Validation."""
    df = df.copy()
    
    # Ensure match_id exists
    if "match_id" not in df.columns:
        # Construct match_id if missing (fallback)
        df["match_id"] = (df["match_date"].astype(str) + "_" + 
                          df["team_id_h"].astype(str) + "_" + 
                          df["team_id_a"].astype(str))
        
    unique_matches = df["match_id"].unique()
    y_true_all = []
    y_pred_all = []
    
    for match_id in unique_matches:
        train_mask = df["match_id"] != match_id
        test_mask = df["match_id"] == match_id
        
        df_train = df[train_mask]
        df_test = df[test_mask]
        
        if df_test.empty:
            continue
            
        # Calculate y_test
        serve_is_home = (df_test["serve_team"] == "h")
        y_test = np.where(
            (serve_is_home & (df_test["point_won_team"] == "h")) |
            (~serve_is_home & (df_test["point_won_team"] == "a")),
            1, 0
        )
        
        # Fit and Predict
        model.fit(df_train)
        y_prob = model.predict_proba(df_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_prob)
        
    return y_true_all, y_pred_all

def run_sequential_backtest(model, df):
    """
    Run Weekly Sequential Backtest.
    Sorts matches by date.
    Iterates through weeks (or matches).
    Trains on all PAST matches, predicts CURRENT match(es).
    """
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], format='mixed', dayfirst=True)
    df = df.sort_values("match_date")
    
    # We need a minimum history to start training.
    # Let's say we need at least 5 matches to start.
    # Or we can just start from the second match.
    
    unique_dates = np.sort(df["match_date"].unique())
    
    y_true_all = []
    y_pred_all = []
    
    # Start from the 5th date to have some training data
    start_idx = 5
    if len(unique_dates) <= start_idx:
        start_idx = 1 # Fallback if very few dates
        
    for i in range(start_idx, len(unique_dates)):
        current_date = unique_dates[i]
        
        # Train on all data strictly BEFORE current_date
        train_mask = df["match_date"] < current_date
        # Test on data ON current_date
        test_mask = df["match_date"] == current_date
        
        df_train = df[train_mask]
        df_test = df[test_mask]
        
        if df_train.empty or df_test.empty:
            continue
            
        # Calculate y_test
        serve_is_home = (df_test["serve_team"] == "h")
        y_test = np.where(
            (serve_is_home & (df_test["point_won_team"] == "h")) |
            (~serve_is_home & (df_test["point_won_team"] == "a")),
            1, 0
        )
        
        # Fit and Predict
        model.fit(df_train)
        y_prob = model.predict_proba(df_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_prob)
        
    return y_true_all, y_pred_all

def plot_calibration(y_true, y_pred, model_name, min_samples_per_bin=400):
    """
    Plot calibration curve with uncertainty.
    Returns the plot as a base64 encoded string (for Streamlit).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Sort by prediction
    sorted_indices = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    n_samples = len(y_pred)
    if n_samples == 0:
        return None
        
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
        extra = 1 if i < remainder else 0
        bin_size = base_size + extra
        end_idx = current_idx + bin_size
        
        bin_y_true = y_true_sorted[current_idx:end_idx]
        bin_y_pred = y_pred_sorted[current_idx:end_idx]
        
        prob_true.append(np.mean(bin_y_true))
        prob_pred.append(np.mean(bin_y_pred))
        bin_counts.append(len(bin_y_true))
        
        current_idx = end_idx
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot range
    all_values = prob_true + prob_pred
    if not all_values:
        plot_min, plot_max = 0, 1
    else:
        plot_min = max(0, min(all_values) - 0.05)
        plot_max = min(1, max(all_values) + 0.05)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect', alpha=0.5)
    
    # Calibration curve with CI
    if prob_pred:
        p_hat = np.array(prob_true)
        ns = np.array(bin_counts)
        p_hat_safe = np.clip(p_hat, 0.01, 0.99)
        se = np.sqrt(p_hat_safe * (1 - p_hat_safe) / ns)
        yerr = 1.96 * se
        
        ax.errorbar(prob_pred, prob_true, yerr=yerr, fmt='none', 
                     ecolor='gray', alpha=0.5, capsize=3)
        
        ax.scatter(prob_pred, prob_true, s=50, alpha=0.7, label=model_name)
        ax.plot(prob_pred, prob_true, '-', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Actual Fraction')
    ax.set_title(f'Calibration: {model_name}')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    # Encode
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"
