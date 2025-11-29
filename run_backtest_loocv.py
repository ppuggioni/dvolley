import pandas as pd
from models import LogisticRegressionModel, EmpiricalModel, SimpleEmpiricalModel, GlobalMeanModel
from backtest_engine import BackTestEngine

def main():
    # Load Data
    csv_path = "clean_data/clean_data.csv"
    try:
        df = pd.read_csv(csv_path, encoding="cp1252")
        print(f"Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Define Models
    models = [
        GlobalMeanModel(),
        SimpleEmpiricalModel(),
        EmpiricalModel(),
        LogisticRegressionModel(alpha=0.1),
        LogisticRegressionModel(alpha=0.01),
        LogisticRegressionModel(alpha=0.001),
        LogisticRegressionModel(alpha=1e-5)
    ]
    
    # Initialize Engine with different output directory
    engine = BackTestEngine(output_dir="backtest_loocv_results")
    
    # Run Leave-One-Match-Out CV
    results = engine.run_leave_one_match_out(df, models)
    
    # Evaluate
    engine.evaluate_results(results)
    
    print("\nLOOCV Backtest Complete. Check 'backtest_loocv_results' directory for plots and summary.")

if __name__ == "__main__":
    main()
