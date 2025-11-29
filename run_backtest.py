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
    
    # Initialize Engine
    engine = BackTestEngine(output_dir="backtest_results")
    
    # Run Sequential Backtest
    results = engine.run_sequential_weekly(df, models)
    
    # Evaluate
    engine.evaluate_results(results)
    
    print("\nBacktest Complete. Check 'backtest_results' directory for plots and summary.")

if __name__ == "__main__":
    main()
