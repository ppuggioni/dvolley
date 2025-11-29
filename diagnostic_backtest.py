import pandas as pd
import numpy as np
from models import LogisticRegressionModel, EmpiricalModel
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    csv_path = "clean_data/clean_data.csv"
    df = pd.read_csv(csv_path, encoding="cp1252")
    
    print("="*80)
    print("DIAGNOSTIC CHECK: Backtest Logic Verification")
    print("="*80)
    
    # Check 1: Date Parsing
    print("\n## CHECK 1: Date Parsing")
    try:
        df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True)
        print(f"✓ Dates parsed successfully with dayfirst=True")
    except Exception as e:
        print(f"✗ Error with dayfirst=True: {e}")
        df["match_date"] = pd.to_datetime(df["match_date"], format='mixed', dayfirst=True)
        print(f"✓ Dates parsed with format='mixed'")
    
    print(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
    print(f"Total rows: {len(df)}")
    
    # Check 2: Week Assignment
    print("\n## CHECK 2: Week Assignment")
    df = df.sort_values("match_date")
    df["week"] = df["match_date"].dt.to_period("W")
    unique_weeks = sorted(df["week"].unique())
    
    print(f"Number of unique weeks: {len(unique_weeks)}")
    print(f"Weeks: {unique_weeks[:5]} ... {unique_weeks[-5:]}")
    
    # Show distribution of matches per week
    week_counts = df.groupby("week").size()
    print(f"\nMatches per week stats:")
    print(f"  Min: {week_counts.min()}, Max: {week_counts.max()}, Mean: {week_counts.mean():.1f}")
    
    # Check 3: Train/Test Split for a Sample Week
    print("\n## CHECK 3: Train/Test Split Logic")
    sample_week_idx = 5  # Pick week 5 (after some training data)
    sample_week = unique_weeks[sample_week_idx]
    
    train_mask = df["week"] < sample_week
    test_mask = df["week"] == sample_week
    
    df_train = df[train_mask]
    df_test = df[test_mask]
    
    print(f"\nSample Week: {sample_week}")
    print(f"Training data size: {len(df_train)} rows")
    print(f"  From weeks: {sorted(df_train['week'].unique())}")
    print(f"Test data size: {len(df_test)} rows")
    print(f"  From week: {sorted(df_test['week'].unique())}")
    
    # Verify no overlap
    train_dates = set(df_train["match_date"])
    test_dates = set(df_test["match_date"])
    overlap = train_dates & test_dates
    print(f"Date overlap: {len(overlap)} (should be 0)")
    
    # Check 4: y calculation consistency
    print("\n## CHECK 4: Target (y) Calculation")
    
    def calc_y(df_subset):
        serve_is_home = (df_subset["serve_team"] == "h")
        y = np.where(
            (serve_is_home & (df_subset["point_won_team"] == "h")) |
            (~serve_is_home & (df_subset["point_won_team"] == "a")),
            1, 0
        )
        return y
    
    y_test = calc_y(df_test)
    print(f"Test set y: mean={y_test.mean():.3f}, sum={y_test.sum()}/{len(y_test)}")
    print(f"  (Should be break point probability, typically 0.3-0.4)")
    
    # Check first few rows manually
    print("\nManual verification of first 5 test rows:")
    for i in range(min(5, len(df_test))):
        row = df_test.iloc[i]
        serve = row["serve_team"]
        won = row["point_won_team"]
        y_val = y_test[i]
        expected = 1 if (serve == won) else 0
        status = "✓" if y_val == expected else "✗"
        print(f"  {status} Row {i}: serve={serve}, won={won}, y={y_val} (expected={expected})")
    
    # Check 5: Model Fit and Predict
    print("\n## CHECK 5: Model Fit and Predict")
    
    # Test LogisticRegression
    print("\n### LogisticRegressionModel (alpha=0.1)")
    model_lr = LogisticRegressionModel(alpha=0.1)
    
    print("Fitting on training data...")
    model_lr.fit(df_train)
    print("✓ Fit complete")
    
    print("Predicting on test data...")
    y_pred_lr = model_lr.predict_proba(df_test)
    print(f"✓ Predictions: min={y_pred_lr.min():.3f}, max={y_pred_lr.max():.3f}, mean={y_pred_lr.mean():.3f}")
    
    # Check if predictions are in reasonable range
    if y_pred_lr.min() < 0 or y_pred_lr.max() > 1:
        print("  ✗ ERROR: Predictions outside [0, 1]!")
    
    # Compare to actual
    from sklearn.metrics import log_loss
    ll = log_loss(y_test, y_pred_lr, labels=[0, 1])
    print(f"  Log Loss: {ll:.4f}")
    
    # Test EmpiricalModel
    print("\n### EmpiricalModel")
    model_emp = EmpiricalModel()
    
    print("Fitting on training data...")
    model_emp.fit(df_train)
    print(f"✓ Fit complete")
    print(f"  Learned {len(model_emp.emp_break_pos)} server position stats")
    print(f"  Learned {len(model_emp.emp_sideout_pos)} receiver position stats")
    
    print("Predicting on test data...")
    y_pred_emp = model_emp.predict_proba(df_test)
    print(f"✓ Predictions: min={y_pred_emp.min():.3f}, max={y_pred_emp.max():.3f}, mean={y_pred_emp.mean():.3f}")
    
    ll_emp = log_loss(y_test, y_pred_emp, labels=[0, 1])
    print(f"  Log Loss: {ll_emp:.4f}")
    
    # Check 6: Verify prediction logic manually for one row
    print("\n## CHECK 6: Manual Prediction Verification")
    row_idx = 0
    row = df_test.iloc[row_idx]
    
    serve_is_home = row["serve_team"] == "h"
    server_id = row["team_id_h"] if serve_is_home else row["team_id_a"]
    receiver_id = row["team_id_a"] if serve_is_home else row["team_id_h"]
    server_pos = row["p_h"] if serve_is_home else row["p_a"]
    receiver_pos = row["p_a"] if serve_is_home else row["p_h"]
    
    print(f"\nTest row {row_idx}:")
    print(f"  Server: {server_id} (pos {server_pos})")
    print(f"  Receiver: {receiver_id} (pos {receiver_pos})")
    print(f"  Actual outcome (y): {y_test[row_idx]}")
    
    # Empirical model prediction
    p_break = model_emp.emp_break_pos.get((str(server_id), server_pos), model_emp.global_mean)
    p_sideout = model_emp.emp_sideout_pos.get((str(receiver_id), receiver_pos), model_emp.global_mean)
    pred_emp_manual = (p_break + p_sideout) / 2.0
    
    print(f"\nEmpirical Model:")
    print(f"  P(break | server={server_id}, pos={server_pos}): {p_break:.3f}")
    print(f"  P(server wins | receiver={receiver_id}, pos={receiver_pos}): {p_sideout:.3f}")
    print(f"  Average: {pred_emp_manual:.3f}")
    print(f"  Model prediction: {y_pred_emp[row_idx]:.3f}")
    print(f"  Match: {'✓' if abs(pred_emp_manual - y_pred_emp[row_idx]) < 0.001 else '✗'}")
    
    # Check 7: Verify encoding issue
    print("\n## CHECK 7: CSV Encoding Test")
    import tempfile
    import os
    
    # Save a sample dataframe to CSV and reload
    sample_df = df_train.head(100)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
        sample_df.to_csv(f, index=False)
        temp_path = f.name
    
    try:
        # Try reading with cp1252 (what the model uses)
        df_reload = pd.read_csv(temp_path, encoding="cp1252")
        print(f"✓ Reloaded {len(df_reload)} rows with cp1252 encoding")
        
        # Check if dates are preserved
        orig_dates = sample_df["match_date"].astype(str).tolist()[:3]
        reload_dates = df_reload["match_date"].astype(str).tolist()[:3]
        print(f"  Original dates: {orig_dates}")
        print(f"  Reloaded dates: {reload_dates}")
        print(f"  Match: {'✓' if orig_dates == reload_dates else '✗'}")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC CHECK COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
