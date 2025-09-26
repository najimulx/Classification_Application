import os
import pandas as pd
from aeroreach.data.data_loader import DataLoader
from aeroreach.preprocessing.preprocessor import DataPreprocessor
from aeroreach.utils.feature_config import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from aeroreach.utils.encoding import Encoder
from aeroreach.clustering.kprototypes_cluster import KPrototypesClustering
from aeroreach.clustering.hierarchical_gower import HierarchicalGowerClustering
from aeroreach.classification.random_forest_classifier import SegmentClassifier
from aeroreach.evaluation.metrics import Evaluation

DATA_PATH = os.path.join(os.path.dirname(__file__), '../AeroReach Insights.csv')

# 1. Load Data
data_loader = DataLoader(DATA_PATH)
df = data_loader.load_data()

# 2. Preprocess Data
preprocessor = DataPreprocessor(CATEGORICAL_COLS, NUMERICAL_COLS)
df = preprocessor.preprocess(df)

# 3. Encode Categorical Variables for clustering (K-Prototypes needs categorical indices)
encoder = Encoder()
df_encoded = encoder.label_encode(df.copy(), CATEGORICAL_COLS)
categorical_indices = [df_encoded.columns.get_loc(col) for col in CATEGORICAL_COLS]

# 4. K-Prototypes Clustering
kproto = KPrototypesClustering(n_clusters=5, categorical_cols=categorical_indices)
kproto_labels = kproto.fit_predict(df_encoded)
df['kproto_cluster'] = kproto_labels

# 5. Hierarchical Clustering with Gower Distance
hier = HierarchicalGowerClustering(n_clusters=5)
hier_labels = hier.fit_predict(df)
df['hier_cluster'] = hier_labels

# 6. Evaluate Clustering
clustering_eval = Evaluation.clustering_scores(df[NUMERICAL_COLS], kproto_labels)
print('K-Prototypes Clustering Evaluation:', clustering_eval)

# 7. Classification Model (Random Forest)
# Use clusters as target for demonstration (replace with actual segment labels as needed)
classifier = SegmentClassifier()
X = df_encoded.drop([TARGET_COL], axis=1)
y = df['kproto_cluster']
X_train, X_val, X_test, y_train, y_val, y_test = classifier.train(X, y)
y_pred = classifier.predict(X_test)

# 8. Evaluate Classification
class_eval = Evaluation.classification_report(y_test, y_pred)
print('Classification Report:', class_eval)
