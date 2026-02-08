from __future__ import annotations

import numpy as np
import pandas as pd

from dvolley.domain.model_base import BaseModel


class EmpiricalModel(BaseModel):
    """Empirical model using team and rotation statistics."""

    def __init__(self):
        self.emp_break_pos = {}  # (team_id, pos) -> prob
        self.emp_sideout_pos = {}  # (team_id, pos) -> prob (receiver wins)
        self.global_mean = 0.5
        self.team_names = {}

    @staticmethod
    def _extract_team_names(df: pd.DataFrame) -> dict:
        team_names = {}
        if "team_id_h" in df.columns and "team_h" in df.columns:
            pairs = df[["team_id_h", "team_h"]].dropna().drop_duplicates()
            for _, row in pairs.iterrows():
                team_names[str(row["team_id_h"])] = str(row["team_h"])
        if "team_id_a" in df.columns and "team_a" in df.columns:
            pairs = df[["team_id_a", "team_a"]].dropna().drop_duplicates()
            for _, row in pairs.iterrows():
                team_names[str(row["team_id_a"])] = str(row["team_a"])
        return team_names

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        self.team_names = self._extract_team_names(df)

        serve_is_home = df["serve_team"] == "h"

        # y = 1 if server wins point
        y = np.where(
            (serve_is_home & (df["point_won_team"] == "h"))
            | (~serve_is_home & (df["point_won_team"] == "a")),
            1,
            0,
        )
        y_receiver = 1 - y

        self.global_mean = y.mean()

        df["y_server"] = y
        df["y_receiver"] = y_receiver
        df["server_id"] = np.where(serve_is_home, df["team_id_h"], df["team_id_a"])
        df["receiver_id"] = np.where(serve_is_home, df["team_id_a"], df["team_id_h"])

        # Server/Receiver rotation
        df["server_pos"] = np.where(serve_is_home, df["p_h"], df["p_a"])
        df["receiver_pos"] = np.where(serve_is_home, df["p_a"], df["p_h"])

        # Calculate means
        self.emp_break_pos = df.groupby(["server_id", "server_pos"])["y_server"].mean().to_dict()
        self.emp_sideout_pos = df.groupby(["receiver_id", "receiver_pos"])["y_receiver"].mean().to_dict()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        serve_is_home = df["serve_team"] == "h"
        server_ids = np.where(serve_is_home, df["team_id_h"], df["team_id_a"])
        receiver_ids = np.where(serve_is_home, df["team_id_a"], df["team_id_h"])

        server_pos = np.where(serve_is_home, df["p_h"], df["p_a"])
        receiver_pos = np.where(serve_is_home, df["p_a"], df["p_h"])

        preds = []
        for i in range(len(df)):
            s_id = str(server_ids[i])
            r_id = str(receiver_ids[i])
            s_p = int(server_pos[i])
            r_p = int(receiver_pos[i])

            p_break = self.emp_break_pos.get((s_id, s_p), self.global_mean)
            p_sideout = self.emp_sideout_pos.get((r_id, r_p), 1 - self.global_mean)

            # Average server strength with receiver weakness
            preds.append((p_break + (1 - p_sideout)) / 2.0)

        return np.array(preds)

    @property
    def name(self) -> str:
        return "empirical_team_rotation"

    def get_simulator_params(self) -> dict:
        return {
            "type": "empirical",
            "break_pos": self.emp_break_pos,
            "sideout_pos": self.emp_sideout_pos,
            "global_mean": self.global_mean,
            "level": "rotation",
            "team_names": self.team_names,
        }


class SimpleEmpiricalModel(BaseModel):
    """Empirical model using only team-level statistics (no rotation info)."""

    def __init__(self):
        self.emp_break_team = {}  # team_id -> prob
        self.emp_sideout_team = {}  # team_id -> prob
        self.global_mean = 0.5
        self.team_names = {}

    @staticmethod
    def _extract_team_names(df: pd.DataFrame) -> dict:
        team_names = {}
        if "team_id_h" in df.columns and "team_h" in df.columns:
            pairs = df[["team_id_h", "team_h"]].dropna().drop_duplicates()
            for _, row in pairs.iterrows():
                team_names[str(row["team_id_h"])] = str(row["team_h"])
        if "team_id_a" in df.columns and "team_a" in df.columns:
            pairs = df[["team_id_a", "team_a"]].dropna().drop_duplicates()
            for _, row in pairs.iterrows():
                team_names[str(row["team_id_a"])] = str(row["team_a"])
        return team_names

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        self.team_names = self._extract_team_names(df)

        serve_is_home = df["serve_team"] == "h"
        y = np.where(
            (serve_is_home & (df["point_won_team"] == "h"))
            | (~serve_is_home & (df["point_won_team"] == "a")),
            1,
            0,
        )
        y_receiver = 1 - y

        self.global_mean = y.mean()

        df["y_server"] = y
        df["y_receiver"] = y_receiver
        df["server_id"] = np.where(serve_is_home, df["team_id_h"], df["team_id_a"])
        df["receiver_id"] = np.where(serve_is_home, df["team_id_a"], df["team_id_h"])

        self.emp_break_team = df.groupby("server_id")["y_server"].mean().to_dict()
        self.emp_sideout_team = df.groupby("receiver_id")["y_receiver"].mean().to_dict()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for col in ["team_id_h", "team_id_a"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        serve_is_home = df["serve_team"] == "h"
        server_ids = np.where(serve_is_home, df["team_id_h"], df["team_id_a"])
        receiver_ids = np.where(serve_is_home, df["team_id_a"], df["team_id_h"])

        preds = []
        for i in range(len(df)):
            s_id = str(server_ids[i])
            r_id = str(receiver_ids[i])

            p_break = self.emp_break_team.get(s_id, self.global_mean)
            p_sideout = self.emp_sideout_team.get(r_id, 1 - self.global_mean)

            preds.append((p_break + (1 - p_sideout)) / 2.0)

        return np.array(preds)

    @property
    def name(self) -> str:
        return "empirical_team"

    def get_simulator_params(self) -> dict:
        return {
            "type": "empirical",
            "break_team": self.emp_break_team,
            "sideout_team": self.emp_sideout_team,
            "global_mean": self.global_mean,
            "level": "team",
            "team_names": self.team_names,
        }


class GlobalMeanModel(BaseModel):
    """Simplest baseline: predicts global mean for all rallies."""

    def __init__(self):
        self.global_mean = 0.5

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        serve_is_home = df["serve_team"] == "h"
        y = np.where(
            (serve_is_home & (df["point_won_team"] == "h"))
            | (~serve_is_home & (df["point_won_team"] == "a")),
            1,
            0,
        )
        self.global_mean = y.mean()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.global_mean)

    @property
    def name(self) -> str:
        return "empirical_global_only"

    def get_simulator_params(self) -> dict:
        return {
            "type": "empirical",
            "global_mean": self.global_mean,
            "level": "global",
        }
