import pandas as pd
import numpy as np
from models import LogisticRegressionModel, EmpiricalModel
from sklearn.metrics import log_loss

# Load data
df = pd.read_csv("clean_data/clean_data.csv", encoding="cp1252")
print("="*60)
print("BACKTEST VERIFICATION")
print("="*60)

# Parse dates
df["match_date"] = pd.to_datetime(df["match_date"], format='mixed', dayfirst=True)
df = df.sort_values("match_date")
df["week"] = df["match_date"].dt.to_period("W")

print(f"\n1. DATA SUMMARY")
print(f"   Total rows: {len(df)}")
print(f"   Date range: {df['match_date'].min()} to {df['match_date'].max()}")
print(f"   Unique weeks: {len(df['week'].unique())}")

# Calculate global break point rate
serve_is_home = (df["serve_team"] == "h")
y_all = np.where(
    (serve_is_home & (df["point_won_team"] == "h")) |
    (~serve_is_home & (df["point_won_team"] == "a")),
    1, 0
)
print(f"   Overall break point rate: {y_all.mean():.3f}")

# Pick a sample week for testing
unique_weeks = sorted(df["week"].unique())
test_week = unique_weeks[5]  # Week 5

train_mask = df["week"] < test_week
test_mask = df["week"] == test_week

df_train = df[train_mask]
df_test = df[test_mask]

print(f"\n2. TRAIN/TEST SPLIT (Week {test_week})")
print(f"   Train: {len(df_train)} rows from {len(df_train['week'].unique())} weeks")
print(f"   Test:  {len(df_test)} rows")

# Calculate test y
y_test = np.where(
    ((df_test["serve_team"] == "h") & (df_test["point_won_team"] == "h")) |
    ((df_test["serve_team"] == "a") & (df_test["point_won_team"] == "a")),
    1, 0
)
print(f"   Test break point rate: {y_test.mean():.3f}")

# Test LogisticRegression
print(f"\n3. LOGISTIC REGRESSION (alpha=0.1)")
model_lr = LogisticRegressionModel(alpha=0.1)
model_lr.fit(df_train)
y_pred_lr = model_lr.predict_proba(df_test)

print(f"   Predictions: min={y_pred_lr.min():.3f}, max={y_pred_lr.max():.3f}, mean={y_pred_lr.mean():.3f}")
print(f"   Log Loss: {log_loss(y_test, y_pred_lr):.4f}")

# Check a few predictions manually
print(f"\n   First 5 predictions vs actuals:")
for i in range(5):
    print(f"      Row {i}: pred={y_pred_lr[i]:.3f}, actual={y_test[i]}")

# Test EmpiricalModel
print(f"\n4. EMPIRICAL MODEL")
model_emp = EmpiricalModel()
model_emp.fit(df_train)
y_pred_emp = model_emp.predict_proba(df_test)

print(f"   Predictions: min={y_pred_emp.min():.3f}, max={y_pred_emp.max():.3f}, mean={y_pred_emp.mean():.3f}")
print(f"   Log Loss: {log_loss(y_test, y_pred_emp):.4f}")

# Manual check of first prediction
row = df_test.iloc[0]
serve_is_home_row = row["serve_team"] == "h"
server_id = str(row["team_id_h"] if serve_is_home_row else row["team_id_a"])
receiver_id = str(row["team_id_a"] if serve_is_home_row else row["team_id_h"])
server_pos = row["p_h"] if serve_is_home_row else row["p_a"]
receiver_pos = row["p_a"] if serve_is_home_row else row["p_h"]

p_break = model_emp.emp_break_pos.get((server_id, server_pos), model_emp.global_mean)
p_sideout = model_emp.emp_sideout_pos.get((receiver_id, receiver_pos), model_emp.global_mean)
pred_manual = (p_break + p_sideout) / 2.0

print(f"\n5. MANUAL PREDICTION CHECK (First test row)")
print(f"   Server: {server_id}, Pos: {server_pos}")
print(f"   P(break): {p_break:.3f}")
print(f"   P(server wins when {receiver_id} receives at {receiver_pos}): {p_sideout:.3f}")
print(f"   Average: {pred_manual:.3f}")
print(f"   Model prediction: {y_pred_emp[0]:.3f}")
print(f"   Match: {abs(pred_manual - y_pred_emp[0]) < 0.001}")

print(f"\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
