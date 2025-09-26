import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from typing import List

class Encoder:
    def __init__(self):
        # store classes for each column as a list
        self.label_encoders = {}  # col -> classes list
        self.onehot_encoder = None

    def label_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Label-encode columns in-place. If an encoder for a column exists, use its mapping.
        Unknown categories are assigned to a new integer (len(classes)).
        """
        for col in cols:
            series = df[col].astype(str)
            if col not in self.label_encoders:
                # fit: store observed classes
                classes = pd.Series(series.unique()).astype(str).tolist()
                self.label_encoders[col] = classes
            else:
                classes = self.label_encoders[col]
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            # map unknowns to len(classes)
            df[col] = series.map(mapping).fillna(len(mapping)).astype(int)
        return df

    def onehot_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded = self.onehot_encoder.fit_transform(df[cols])
        encoded_df = pd.DataFrame(encoded, columns=self.onehot_encoder.get_feature_names_out(cols))
        df = df.drop(cols, axis=1)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return df
