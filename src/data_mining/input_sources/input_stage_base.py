from abc import ABC, abstractmethod
import pandas as pd

class InputStage(ABC):
    """Input stage interface for plugin-based data loading"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data from the input source and return a DataFrame"""
        raise NotImplementedError
