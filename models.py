import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from analysis_regr import VolleyballBreakpointSideoutRegModelNoHome

class BaseModel(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns probability of server winning (break point)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class LogisticRegressionModel(BaseModel):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.model = VolleyballBreakpointSideoutRegModelNoHome(alpha=alpha)
        
    def fit(self, df: pd.DataFrame):
        # The model class expects to load data from CSV usually, but we can hack it 
        # or we can modify it to accept a dataframe. 
        # Looking at analysis_regr.py, load_data takes a csv path.
        # But we can manually set the internal state if we are careful, 
        # OR we can save to a temp csv. Saving to temp csv is safer/easier given existing code.
        
        # Actually, let's see if we can just inject the dataframe.
        # The load_data method does a lot of preprocessing (indices, weights, etc).
        # It's better to modify the underlying class to accept a DF, but for now 
        # I will use a temp file approach to be robust and not break existing code.
        
        # Wait, I can just copy the logic from load_data but accept a DF.
        # But to avoid code duplication, I'll use the temp file trick for now.
        # It's not the most efficient but it's robust.
        
        import tempfile
        import os
        
        # Create temp csv
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            df.to_csv(f, index=False)
            temp_path = f.name
            
        try:
            self.model.load_data(temp_path)
            self.model.fit()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba_breakpoint(df)
        
    @property
    def name(self) -> str:
        return f"LogReg(alpha={self.alpha})"

class EmpiricalModel(BaseModel):
    def __init__(self):
        self.emp_break_pos = {} # team_id -> pos -> prob
        self.emp_sideout_pos = {} # team_id -> pos -> prob
        self.global_mean = 0.5
        
    def fit(self, df: pd.DataFrame):
        # Calculate empirical probabilities
        # We need to replicate the logic:
        # Break point prob for (Team T, Rotation R) = Mean of points won when T serves in R
        # Sideout prob for (Team T, Rotation R) = Mean of points won when T receives in R (which is 1 - server_win)
        # Wait, the user defined Empirical as: "avg of the two empirical probabilities"
        # In compare_alphas.py: (t1_bp_pos_emp[r] + t2_so_pos_emp[c]) / 2.0
        # where t1_bp_pos_emp is mean of server winning
        # and t2_so_pos_emp is mean of SERVER winning when T2 is receiving?
        
        # Let's check analysis_regr.py _compute_empirical_buckets:
        # emp_sideout_pos[t] = y[mask_r].mean()
        # y is 1 if server wins.
        # So emp_sideout_pos[t] is "Prob server wins when T receives".
        # So yes, we just average them.
        
        # Preprocessing
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
        # Identify server/receiver for each row
        serve_is_home = (df["serve_team"] == "h")
        
        # We need to handle team IDs carefully
        # Let's just iterate over unique teams
        teams = pd.concat([df["team_id_h"], df["team_id_a"]]).unique()
        
        # Point won by server?
        # point_won_team = h/a
        # server = h if serve_is_home else a
        # y = 1 if server == point_won_team
        
        y = np.where(
            (serve_is_home & (df["point_won_team"] == "h")) |
            (~serve_is_home & (df["point_won_team"] == "a")),
            1, 0
        )
        
        self.global_mean = y.mean()
        
        # Store data for fast lookup
        # We can use groupby
        
        # Add helper columns
        df['y'] = y
        df['server_id'] = np.where(serve_is_home, df['team_id_h'], df['team_id_a'])
        df['receiver_id'] = np.where(serve_is_home, df['team_id_a'], df['team_id_h'])
        
        # Server rotation (1-6)
        # if serve_is_home, p_h; else p_a
        df['server_pos'] = np.where(serve_is_home, df['p_h'], df['p_a'])
        
        # Receiver rotation (1-6)
        # if serve_is_home, p_a; else p_h
        df['receiver_pos'] = np.where(serve_is_home, df['p_a'], df['p_h'])
        
        # Groupby
        self.emp_break_pos = df.groupby(['server_id', 'server_pos'])['y'].mean().to_dict()
        self.emp_sideout_pos = df.groupby(['receiver_id', 'receiver_pos'])['y'].mean().to_dict()
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # For each row, look up server stats and receiver stats
        
        # We need to do this efficiently. Apply is slow but easiest to write.
        # Vectorized map is better.
        
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
        serve_is_home = (df["serve_team"] == "h")
        server_ids = np.where(serve_is_home, df['team_id_h'], df['team_id_a'])
        receiver_ids = np.where(serve_is_home, df['team_id_a'], df['team_id_h'])
        
        server_pos = np.where(serve_is_home, df['p_h'], df['p_a'])
        receiver_pos = np.where(serve_is_home, df['p_a'], df['p_h'])
        
        preds = []
        for i in range(len(df)):
            s_id = server_ids[i]
            r_id = receiver_ids[i]
            s_p = server_pos[i]
            r_p = receiver_pos[i]
            
            p_break = self.emp_break_pos.get((s_id, s_p), self.global_mean)
            p_sideout = self.emp_sideout_pos.get((r_id, r_p), self.global_mean)
            
            # Average them
            preds.append((p_break + p_sideout) / 2.0)
            
        return np.array(preds)

    @property
    def name(self) -> str:
        return "EmpiricalModel"

class SimpleEmpiricalModel(BaseModel):
    """Empirical model using only team-level statistics (no rotation info)."""
    
    def __init__(self):
        self.emp_break_team = {}  # team_id -> prob
        self.emp_sideout_team = {}  # team_id -> prob
        self.global_mean = 0.5
        
    def fit(self, df: pd.DataFrame):
        # Calculate empirical probabilities at team level only
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
        # Identify server/receiver for each row
        serve_is_home = (df["serve_team"] == "h")
        
        # Point won by server?
        y = np.where(
            (serve_is_home & (df["point_won_team"] == "h")) |
            (~serve_is_home & (df["point_won_team"] == "a")),
            1, 0
        )
        
        self.global_mean = y.mean()
        
        # Add helper columns
        df['y'] = y
        df['server_id'] = np.where(serve_is_home, df['team_id_h'], df['team_id_a'])
        df['receiver_id'] = np.where(serve_is_home, df['team_id_a'], df['team_id_h'])
        
        # Groupby team only (no position)
        self.emp_break_team = df.groupby('server_id')['y'].mean().to_dict()
        self.emp_sideout_team = df.groupby('receiver_id')['y'].mean().to_dict()
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
        serve_is_home = (df["serve_team"] == "h")
        server_ids = np.where(serve_is_home, df['team_id_h'], df['team_id_a'])
        receiver_ids = np.where(serve_is_home, df['team_id_a'], df['team_id_h'])
        
        preds = []
        for i in range(len(df)):
            s_id = server_ids[i]
            r_id = receiver_ids[i]
            
            p_break = self.emp_break_team.get(s_id, self.global_mean)
            p_sideout = self.emp_sideout_team.get(r_id, self.global_mean)
            
            # Average them
            preds.append((p_break + p_sideout) / 2.0)
            
        return np.array(preds)

    @property
    def name(self) -> str:
        return "SimpleEmpirical"

class GlobalMeanModel(BaseModel):
    """Simplest baseline: predicts global mean for all rallies."""
    
    def __init__(self):
        self.global_mean = 0.5
        
    def fit(self, df: pd.DataFrame):
        # Calculate global break point probability
        df = df.copy()
        
        serve_is_home = (df["serve_team"] == "h")
        y = np.where(
            (serve_is_home & (df["point_won_team"] == "h")) |
            (~serve_is_home & (df["point_won_team"] == "a")),
            1, 0
        )
        
        self.global_mean = y.mean()
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Return the same probability for all rows
        return np.full(len(df), self.global_mean)

    @property
    def name(self) -> str:
        return "GlobalMean"

