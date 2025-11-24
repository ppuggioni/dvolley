import pandas as pd
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    # 1. Load Data
    params_path = "params/params_out_break_sideout.csv"
    try:
        df = pd.read_csv(params_path)
    except FileNotFoundError:
        print(f"Error: Could not find {params_path}")
        return

    # 2. Select Teams
    # Team 1 (Server): Belluno Volley (6727)
    # Team 2 (Receiver): CUS Cagliari (6728)
    team1_id = 6727
    team2_id = 6728
    
    # Get Team Names for display
    t1_name = df[df['team_id'] == str(team1_id)]['team_name'].iloc[0] if not df[df['team_id'] == str(team1_id)].empty else "Team 1"
    t2_name = df[df['team_id'] == str(team2_id)]['team_name'].iloc[0] if not df[df['team_id'] == str(team2_id)].empty else "Team 2"

    print(f"Comparing Break Point Probabilities:")
    print(f"Server (Rows): {t1_name} ({team1_id})")
    print(f"Receiver (Cols): {t2_name} ({team2_id})")
    print("-" * 30)

    # 3. Extract Parameters
    
    # Global
    global_row = df[df['par_name'] == 'global_breakpoint']
    if global_row.empty:
        print("Error: global_breakpoint not found")
        return
    global_intercept = float(global_row['par_value'].iloc[0])

    # Team 1 (Server) Parameters
    t1_df = df[df['team_id'] == str(team1_id)]
    
    t1_bp_adj = float(t1_df[t1_df['par_name'] == 'breakpoint_team_adjustment']['par_value'].iloc[0])
    
    t1_bp_pos_params = {}
    t1_bp_pos_empirical = {}
    
    for i in range(1, 7):
        row = t1_df[t1_df['par_name'] == f'breakpoint_pos_{i}']
        t1_bp_pos_params[i] = float(row['par_value'].iloc[0])
        t1_bp_pos_empirical[i] = float(row['empirical_probability'].iloc[0])

    # Team 2 (Receiver) Parameters
    t2_df = df[df['team_id'] == str(team2_id)]
    
    t2_so_adj = float(t2_df[t2_df['par_name'] == 'sideout_team_adjustment']['par_value'].iloc[0])
    
    t2_so_pos_params = {}
    t2_so_pos_empirical = {} # Note: This is the probability of the server scoring a BP when T2 is receiving in this rotation
    
    for i in range(1, 7):
        row = t2_df[t2_df['par_name'] == f'sideout_pos_{i}']
        t2_so_pos_params[i] = float(row['par_value'].iloc[0])
        # The empirical probability in the sideout rows is the probability of a Break Point happening?
        # Let's check the user request: "note that it's the prob that a bp happens, so if it's the sideout_pos_1 parameter, the lower the empirical prob the better for that team!"
        # Yes, so this value is P(BP).
        t2_so_pos_empirical[i] = float(row['empirical_probability'].iloc[0])

    # 4. Calculate Matrices
    
    # Rows: T1 Rotation 1-6
    # Cols: T2 Rotation 1-6
    
    model_matrix = np.zeros((6, 6))
    empirical_matrix = np.zeros((6, 6))

    for r in range(1, 7): # T1 Rotation (Server)
        for c in range(1, 7): # T2 Rotation (Receiver)
            
            # --- Model Calculation ---
            # logit = Global + T1_BP_Adj + T1_BP_Pos[r] - T2_SO_Adj - T2_SO_Pos[c]
            logit = global_intercept + t1_bp_adj + t1_bp_pos_params[r] - t2_so_adj - t2_so_pos_params[c]
            prob_model = sigmoid(logit)
            model_matrix[r-1, c-1] = prob_model
            
            # --- Empirical Calculation ---
            # Avg of:
            # 1. T1's empirical BP prob in rotation r
            # 2. T2's empirical "Conceded BP" prob in rotation c (which is stored directly as empirical_probability in sideout rows)
            prob_empirical = (t1_bp_pos_empirical[r] + t2_so_pos_empirical[c]) / 2.0
            empirical_matrix[r-1, c-1] = prob_empirical

    # 5. Output
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Set float format to display as percentage with 1 decimal place
    pd.options.display.float_format = '{:.1f}%'.format

    cols = [f"T2_P{i}" for i in range(1, 7)]
    rows = [f"T1_P{i}" for i in range(1, 7)]

    # Multiply by 100 for percentage representation if not using format string that handles it?
    # Actually '{:.1f}%'.format(0.377) -> '0.4%' which is wrong. It expects the number to be 0.377.
    # Wait, '{:.1%}'.format(0.377) -> '37.7%'.
    # So I should use '{:.1%}'
    
    pd.options.display.float_format = '{:.1%}'.format

    df_model = pd.DataFrame(model_matrix, index=rows, columns=cols)
    df_empirical = pd.DataFrame(empirical_matrix, index=rows, columns=cols)
    df_diff = df_model - df_empirical

    print("\nMatrix 1: Model Probabilities (Logit)")
    print(df_model)

    print("\nMatrix 2: Empirical Probabilities (Avg)")
    print(df_empirical)

    print("\nDifference (Model - Empirical)")
    print(df_diff)

if __name__ == "__main__":
    main()
