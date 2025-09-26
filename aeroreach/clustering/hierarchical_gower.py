import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from typing import List
import gower

class HierarchicalGowerClustering:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.linkage_matrix = None
        self.labels_ = None

    def fit_predict(self, data: pd.DataFrame) -> List[int]:
        # Compute Gower distance
        gower_dist = gower.gower_matrix(data)
        self.linkage_matrix = linkage(gower_dist, method='ward')
        self.labels_ = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')
        return self.labels_

    def plot_dendrogram(self):
        if self.linkage_matrix is not None:
            dendrogram(self.linkage_matrix)
