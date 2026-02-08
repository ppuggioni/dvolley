from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Returns probability of server winning (break point)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_simulator_params(self) -> dict:
        """Returns parameters needed by the simulator."""
        raise NotImplementedError
