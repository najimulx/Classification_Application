from sklearn.metrics import silhouette_score, davies_bouldin_score, classification_report
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

class Evaluation:
    @staticmethod
    def clustering_scores(X, labels):
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        return {'silhouette_score': sil, 'davies_bouldin_score': db}

    @staticmethod
    def classification_report(y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True
        ))
        
        # Update layout
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=500,
            height=400
        )
        
        return fig

    @staticmethod
    def feature_importance_plot(importances, feature_names, top_n=10):
        df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df = df.sort_values('importance', ascending=False).head(top_n)
        
        fig = px.bar(df, 
                    x='importance', 
                    y='feature',
                    title='Top Feature Importances',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    orientation='h')
        
        fig.update_layout(
            showlegend=False,
            height=400,
            width=700,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
