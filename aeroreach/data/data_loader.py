import pandas as pd
from typing import Optional

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.filepath)
        return self.data

    def get_data(self) -> pd.DataFrame:
        if self.data is None:
            return self.load_data()
        return self.data
