import pandas as pd
import numpy as np
from analysis_regr import VolleyballBreakpointSideoutRegModelNoHome

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calculate_matrix(params_df, team1_id, team2_id):
    # Global
    global_row = params_df[params_df['par_name'] == 'global_breakpoint']
    if global_row.empty:
        return None
    global_intercept = float(global_row['par_value'].iloc[0])

    # Team 1 (Server)
    t1_df = params_df[params_df['team_id'] == str(team1_id)]
    if t1_df.empty:
        return None
    
    t1_bp_adj = float(t1_df[t1_df['par_name'] == 'breakpoint_team_adjustment']['par_value'].iloc[0])
    t1_bp_pos = {}
    for i in range(1, 7):
        t1_bp_pos[i] = float(t1_df[t1_df['par_name'] == f'breakpoint_pos_{i}']['par_value'].iloc[0])

    # Team 2 (Receiver)
    t2_df = params_df[params_df['team_id'] == str(team2_id)]
    if t2_df.empty:
        return None
        
    t2_so_adj = float(t2_df[t2_df['par_name'] == 'sideout_team_adjustment']['par_value'].iloc[0])
    t2_so_pos = {}
    for i in range(1, 7):
        t2_so_pos[i] = float(t2_df[t2_df['par_name'] == f'sideout_pos_{i}']['par_value'].iloc[0])

    # Matrix
    matrix = np.zeros((6, 6))
    for r in range(1, 7):
        for c in range(1, 7):
            logit = global_intercept + t1_bp_adj + t1_bp_pos[r] - t2_so_adj - t2_so_pos[c]
            matrix[r-1, c-1] = sigmoid(logit)
            
    return matrix

def main():
    alphas = [1e-2, 5e-3, 1e-3, 1e-4, 1e-5]
    csv_path = "clean_data/clean_data.csv"
    
    # Teams
    team1_id = 6727 # Belluno
    team2_id = 6728 # CUS Cagliari
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:.1%}'.format
    
    cols = [f"T2_P{i}" for i in range(1, 7)]
    rows = [f"T1_P{i}" for i in range(1, 7)]

    print(f"Comparing Alphas for {team1_id} (Server) vs {team2_id} (Receiver)")
    print("=" * 60)

    # Calculate Empirical Matrix (Constant)
    # Need to get empirical params from the dataframe (loaded by model)
    # We can just use the last fitted model's params_df since empirical values are constant
    # But to be safe, let's extract them from the first run
    
    empirical_matrix = None
    
    for alpha in [1e-2, 1e-3, 1e-4]:
        print(f"\nFitting with Alpha = {alpha}")
        
        # Instantiate and Fit
        model = VolleyballBreakpointSideoutRegModelNoHome(alpha=alpha)
        model.load_data(csv_path)
        model.fit()
        
        # Get Params
        params_df = model.viz_parameters()
        
        # Calculate Empirical Matrix (only once)
        if empirical_matrix is None:
            # Extract empirical values
            t1_df = params_df[params_df['team_id'] == str(team1_id)]
            t2_df = params_df[params_df['team_id'] == str(team2_id)]
            
            t1_bp_pos_emp = {}
            for i in range(1, 7):
                t1_bp_pos_emp[i] = float(t1_df[t1_df['par_name'] == f'breakpoint_pos_{i}']['empirical_probability'].iloc[0])
                
            t2_so_pos_emp = {}
            for i in range(1, 7):
                t2_so_pos_emp[i] = float(t2_df[t2_df['par_name'] == f'sideout_pos_{i}']['empirical_probability'].iloc[0])
            
            empirical_matrix = np.zeros((6, 6))
            for r in range(1, 7):
                for c in range(1, 7):
                    empirical_matrix[r-1, c-1] = (t1_bp_pos_emp[r] + t2_so_pos_emp[c]) / 2.0
            
            df_emp = pd.DataFrame(empirical_matrix, index=rows, columns=cols)
            print("\nMatrix: Empirical Probabilities (Avg)")
            print(df_emp)

        # Calculate Model Matrix
        matrix = calculate_matrix(params_df, team1_id, team2_id)
        
        if matrix is not None:
            df_model = pd.DataFrame(matrix, index=rows, columns=cols)
            df_diff = df_model - df_emp
            
            print(f"\nMatrix: Model Probabilities (Alpha={alpha})")
            print(df_model)
            print(f"\nDifference (Model - Empirical) (Alpha={alpha})")
            print(df_diff)
        else:
            print("Error calculating matrix (teams not found?)")

if __name__ == "__main__":
    main()
