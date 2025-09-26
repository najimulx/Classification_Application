import pandas as pd
from kmodes.kprototypes import KPrototypes
from typing import List, Tuple

class KPrototypesClustering:
    def __init__(self, n_clusters: int, categorical_cols: List[int]):
        self.n_clusters = n_clusters
        self.categorical_cols = categorical_cols
        self.model = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=5, verbose=1)
        self.labels_ = None

    def fit_predict(self, data: pd.DataFrame) -> List[int]:
        # KPrototypes expects numpy array, categorical columns as indices
        self.labels_ = self.model.fit_predict(data, categorical=self.categorical_cols)
        return self.labels_

    def get_cluster_centroids(self):
        return self.model.cluster_centroids_
