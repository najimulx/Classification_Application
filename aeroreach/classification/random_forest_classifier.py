import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Tuple, Dict

class SegmentClassifier:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        self.feature_importance_dict = {}
        self.cv_scores = None
        self.test_metrics = None

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3, random_state: int = 42) -> Tuple:
        """
        Splits data into train/val/test and fits the model. test_size is fraction for (val+test).
        Performs cross-validation and calculates feature importance.
        Returns (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        self.cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Calculate feature importance
        importances = self.model.feature_importances_
        self.feature_importance_dict = dict(zip(X.columns, importances))
        
        # Calculate test metrics
        y_pred = self.model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        
        self.test_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std()
        }
        
        return (X_train, X_val, X_test, y_train, y_val, y_test)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class and return prediction probabilities.
        Returns (predictions, probabilities)
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return predictions, probabilities

    def predict_with_confidence(self, X: pd.DataFrame) -> Dict:
        """
        Predict class and return comprehensive prediction info.
        """
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        # Get top 2 classes and their probabilities for each prediction
        top2_idx = np.argsort(probabilities, axis=1)[:, -2:]
        top2_classes = self.model.classes_[top2_idx]
        top2_probs = np.take_along_axis(probabilities, top2_idx, axis=1)
        
        results = []
        for i in range(len(predictions)):
            result = {
                'predicted_class': predictions[i],
                'confidence': confidence_scores[i],
                'top_classes': {
                    str(top2_classes[i][1]): float(top2_probs[i][1]),
                    str(top2_classes[i][0]): float(top2_probs[i][0])
                }
            }
            results.append(result)
        
        return results

    def get_model_metrics(self) -> Dict:
        """
        Return comprehensive model metrics.
        """
        if not self.test_metrics:
            return None
        return {
            'test_metrics': self.test_metrics,
            'cross_validation': {
                'mean': float(self.cv_scores.mean()),
                'std': float(self.cv_scores.std()),
                'scores': self.cv_scores.tolist()
            }
        }

    def feature_importances(self):
        """
        Return feature importance scores.
        """
        return self.feature_importance_dict
