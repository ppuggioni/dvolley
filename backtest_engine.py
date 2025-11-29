import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, brier_score_loss
from typing import List, Dict
import os
from models import BaseModel

class BackTestEngine:
    def __init__(self, output_dir="backtest_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def run_sequential_weekly(self, df: pd.DataFrame, models: List[BaseModel]):
        """
        Runs a sequential backtest week by week.
        """
        print(f"Starting Sequential Weekly Backtest on {len(df)} rows...")
        
        # Ensure date is datetime
        df = df.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
            
        # Sort by date
        df = df.sort_values("match_date")
        
        # Add week identifier
        df["week"] = df["match_date"].dt.to_period("W")
        unique_weeks = df["week"].unique()
        unique_weeks = sorted(unique_weeks)
        
        print(f"Found {len(unique_weeks)} unique weeks: {unique_weeks}")
        
        results = {model.name: {"y_true": [], "y_pred": [], "dates": [], "weeks": []} for model in models}
        
        # Iterate through weeks
        # We start from the 2nd week so we have at least some training data
        for i, week in enumerate(unique_weeks):
            if i == 0:
                print(f"Skipping week {week} (no training data)")
                continue
                
            print(f"Processing week {week}...")
            
            # Split data
            train_mask = df["week"] < week
            test_mask = df["week"] == week
            
            df_train = df[train_mask]
            df_test = df[test_mask]
            
            if len(df_train) == 0:
                print("  No training data, skipping.")
                continue
                
            if len(df_test) == 0:
                print("  No test data, skipping.")
                continue
                
            # Calculate y_test for evaluation
            serve_is_home = (df_test["serve_team"] == "h")
            y_test = np.where(
                (serve_is_home & (df_test["point_won_team"] == "h")) |
                (~serve_is_home & (df_test["point_won_team"] == "a")),
                1, 0
            )
            
            # Train and Predict for each model
            for model in models:
                # Fit on past data
                model.fit(df_train)
                
                # Predict on current week
                y_prob = model.predict_proba(df_test)
                
                # Store results
                results[model.name]["y_true"].extend(y_test)
                results[model.name]["y_pred"].extend(y_prob)
                results[model.name]["dates"].extend(df_test["match_date"])
                results[model.name]["weeks"].extend([str(week)] * len(y_test))
                
        return results
    
    def run_leave_one_match_out(self, df: pd.DataFrame, models: List[BaseModel]):
        """
        Leave-One-Match-Out Cross-Validation.
        For each match, train on all other matches and predict on that match.
        """
        print(f"Starting Leave-One-Match-Out CV on {len(df)} rows...")
        
        # Ensure date is datetime
        df = df.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date")
        
        # Identify unique matches by (date, home_team, away_team)
        df["match_id"] = df["match_date"].astype(str) + "_" + df["team_id_h"].astype(str) + "_" + df["team_id_a"].astype(str)
        unique_matches = df["match_id"].unique()
        
        print(f"Found {len(unique_matches)} unique matches")
        
        results = {model.name: {"y_true": [], "y_pred": [], "dates": [], "weeks": []} for model in models}
        
        # Iterate through matches
        for i, match_id in enumerate(unique_matches):
            if (i + 1) % 5 == 0:
                print(f"Processing match {i+1}/{len(unique_matches)}...")
            
            # Split data
            train_mask = df["match_id"] != match_id
            test_mask = df["match_id"] == match_id
            
            df_train = df[train_mask]
            df_test = df[test_mask]
            
            if len(df_train) == 0 or len(df_test) == 0:
                continue
            
            # Calculate y_test
            serve_is_home = (df_test["serve_team"] == "h")
            y_test = np.where(
                (serve_is_home & (df_test["point_won_team"] == "h")) |
                (~serve_is_home & (df_test["point_won_team"] == "a")),
                1, 0
            )
            
            # Train and Predict for each model
            for model in models:
                model.fit(df_train)
                y_prob = model.predict_proba(df_test)
                
                # Store results
                results[model.name]["y_true"].extend(y_test)
                results[model.name]["y_pred"].extend(y_prob)
                results[model.name]["dates"].extend(df_test["match_date"])
                # For LOOCV, we can still track weeks if needed
                if "week" in df_test.columns:
                    results[model.name]["weeks"].extend(df_test["week"].astype(str))
                else:
                    results[model.name]["weeks"].extend([str(df_test["match_date"].iloc[0])] * len(y_test))
                
        return results
        
    def evaluate_results(self, results: Dict):
        """
        Calculates metrics and generates plots.
        """
        summary = []
        weekly_metrics = {model: {"weeks": [], "log_loss": [], "accuracy": [], "samples": []} for model in results}
        
        for model_name, data in results.items():
            y_true = np.array(data["y_true"])
            y_pred = np.array(data["y_pred"])
            weeks = np.array(data["weeks"])
            
            if len(y_true) == 0:
                print(f"No predictions for {model_name}")
                continue
            
            # Overall Metrics
            ll = log_loss(y_true, y_pred, labels=[0, 1])
            bs = brier_score_loss(y_true, y_pred)
            acc = np.mean((y_pred > 0.5) == y_true)
            
            summary.append({
                "Model": model_name,
                "Log Loss": ll,
                "Brier Score": bs,
                "Accuracy": acc,
                "Samples": len(y_true)
            })
            
            # Weekly Metrics
            unique_weeks = sorted(list(set(weeks)))
            for w in unique_weeks:
                mask = weeks == w
                if np.sum(mask) > 0:
                    w_true = y_true[mask]
                    w_pred = y_pred[mask]
                    w_ll = log_loss(w_true, w_pred, labels=[0, 1])
                    w_acc = np.mean((w_pred > 0.5) == w_true)
                    
                    weekly_metrics[model_name]["weeks"].append(w)
                    weekly_metrics[model_name]["log_loss"].append(w_ll)
                    weekly_metrics[model_name]["accuracy"].append(w_acc)
                    weekly_metrics[model_name]["samples"].append(len(w_true))

            # Plot Calibration (Time-Hue)
            self.plot_calibration_time_hue(y_true, y_pred, weeks, model_name)
            
        summary_df = pd.DataFrame(summary)
        print("\nBacktest Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv(os.path.join(self.output_dir, "summary_metrics.csv"), index=False)
        
        # Plot Metrics Time Series
        self.plot_metrics_timeseries(weekly_metrics)
        
        # Plot Probability Comparisons
        self.plot_probability_comparisons(results)
        
        return summary_df
    
    def plot_probability_comparisons(self, results: Dict):
        """Create pairplot comparing predictions across models."""
        
        # Build dataframe with all model predictions
        plot_data = {}
        y_true = None
        
        for model_name, data in results.items():
            if len(data["y_true"]) == 0:
                continue
            plot_data[model_name] = np.array(data["y_pred"])
            if y_true is None:
                y_true = np.array(data["y_true"])
        
        # Create dataframe
        df_plot = pd.DataFrame(plot_data)
        df_plot["Actual"] = y_true
        
        # Pairplot
        g = sns.pairplot(
            df_plot,
            diag_kind="kde",
            plot_kws={"alpha": 0.3, "s": 10},
            corner=True
        )
        g.fig.suptitle("Model Prediction Comparisons", y=1.02, fontsize=16)
        
        plt.savefig(os.path.join(self.output_dir, "prediction_pairplot.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Alternative: heatmap of correlations
        plt.figure(figsize=(10, 8))
        corr = df_plot.corr()
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5,
                   vmin=0, vmax=1, square=True)
        plt.title("Prediction Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "prediction_correlations.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nPrediction comparison plots saved.")
        print(f"Correlation with actual outcomes:")
        for model in plot_data.keys():
            corr_val = np.corrcoef(plot_data[model], y_true)[0, 1]
            print(f"  {model}: {corr_val:.4f}")


    def plot_metrics_timeseries(self, weekly_metrics):
        # Plot Log Loss
        plt.figure(figsize=(12, 6))
        for model_name, data in weekly_metrics.items():
            if not data["weeks"]: continue
            plt.plot(data["weeks"], data["log_loss"], marker='o', label=model_name)
        
        plt.title("Weekly Log Loss (Lower is Better)")
        plt.xlabel("Week")
        plt.ylabel("Log Loss")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "timeseries_log_loss.png"))
        plt.close()

        # Plot Accuracy
        plt.figure(figsize=(12, 6))
        for model_name, data in weekly_metrics.items():
            if not data["weeks"]: continue
            plt.plot(data["weeks"], data["accuracy"], marker='o', label=model_name)
            
        plt.title("Weekly Accuracy (Higher is Better)")
        plt.xlabel("Week")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "timeseries_accuracy.png"))
        plt.close()

    def plot_calibration_time_hue(self, y_true, y_pred, weeks, model_name):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        weeks = np.array(weeks)
        unique_weeks = sorted(list(set(weeks)))
        
        # Setup colormap
        import matplotlib.cm as cm
        colors = cm.viridis(np.linspace(0, 1, len(unique_weeks)))
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect", alpha=0.5)
        
        # Plot each week
        for i, w in enumerate(unique_weeks):
            mask = weeks == w
            if np.sum(mask) < 5: # Skip weeks with too few samples for calibration
                continue
                
            w_true = y_true[mask]
            w_pred = y_pred[mask]
            
            # Simple binning for this week
            # Since samples per week might be small, we use fewer bins (e.g., 5)
            bins = np.linspace(0, 1, 6)
            bin_indices = np.digitize(w_pred, bins)
            
            prob_true = []
            prob_pred = []
            
            for b in range(1, len(bins)):
                b_mask = bin_indices == b
                if np.any(b_mask):
                    prob_true.append(np.mean(w_true[b_mask]))
                    prob_pred.append(np.mean(w_pred[b_mask]))
            
            if prob_pred:
                plt.plot(prob_pred, prob_true, "o-", color=colors[i], label=f"{w}", alpha=0.8)
        
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration by Week: {model_name}")
        # Legend might be too big if many weeks, so maybe put outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Week")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"calibration_time_{model_name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
