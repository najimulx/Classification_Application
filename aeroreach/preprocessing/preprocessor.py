import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, categorical_cols: List[str], numerical_cols: List[str]):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.scaler = StandardScaler()

    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert all numerical columns to numeric, coerce errors to NaN
        for col in self.numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_cols:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else 'Not Available')
        return df

    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        # IQR capping for numerical columns
        for col in self.numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fill_missing(df)
        df = self.cap_outliers(df)
        df = self.scale_features(df)
        return df
