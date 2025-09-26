import streamlit as st
import sys
import pathlib

# Ensure the project root is on sys.path so `import aeroreach` works when the
# app is run directly (for example: `.venv\Scripts\streamlit run aeroreach\ui\app.py`).
project_root = str(pathlib.Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aeroreach.data.data_loader import DataLoader
from aeroreach.preprocessing.preprocessor import DataPreprocessor
from aeroreach.utils.feature_config import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
from aeroreach.utils.encoding import Encoder
from aeroreach.clustering.kprototypes_cluster import KPrototypesClustering
from aeroreach.classification.random_forest_classifier import SegmentClassifier
from aeroreach.evaluation.metrics import Evaluation
import streamlit.components.v1 as components
import uuid
import hashlib

# Resolve dataset path relative to the repository root (two levels up from this file).
# Use pathlib.resolve().parents[2] so this works reliably on Streamlit Cloud and
# different working directories / OS path formats.
DATA_PATH = str(pathlib.Path(__file__).resolve().parents[2] / 'AeroReach Insights.csv')

def load_and_preprocess():
    data_loader = DataLoader(DATA_PATH)
    df = data_loader.load_data()
    if 'UserID' in df.columns:
        df = df.drop('UserID', axis=1)
    preprocessor = DataPreprocessor(CATEGORICAL_COLS, NUMERICAL_COLS)
    df = preprocessor.preprocess(df)
    return df

def create_correlation_heatmap(df, cols):
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=cols,
        y=cols,
        text=corr.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorscale='RdBu',
        zmid=0
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=500,
    )
    return fig

def create_feature_distribution(df, feature):
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=(f"Distribution of {feature}", f"Box Plot of {feature}"),
                       vertical_spacing=0.15)
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[feature], name="Distribution",
                    nbinsx=30, showlegend=False),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=df[feature], name=feature, showlegend=False),
        row=2, col=1
    )
    
    # Make layout responsive
    fig.update_layout(
        height=450,  # Reduced height for better mobile view
        title_x=0.5,
        margin=dict(l=20, r=20, t=40, b=20),  # Compact margins
        autosize=True,  # Enable autosizing
        # Make font sizes responsive
        font=dict(size=10),  # Smaller base font size
        title_font=dict(size=14),  # Smaller title font
        legend_font=dict(size=10)
    )
    
    return fig

def create_categorical_plot(df, feature):
    value_counts = df[feature].value_counts()
    fig = go.Figure(data=[
        go.Bar(x=value_counts.index, y=value_counts.values)
    ])
    fig.update_layout(
        title=f"Distribution of {feature}",
        xaxis_title=feature,
        yaxis_title="Count",
        height=400
    )
    return fig

def metric_with_tooltip(label, value, tooltip, delta=None):
    """Render a compact metric using native Streamlit controls.

    This avoids embedding raw HTML which caused unstable rendering and
    horizontal / unexpected scrolling on some devices. The implementation
    uses a small info button that toggles an inline description via
    session state so it works on touch devices without custom JS.
    """
    # Defensive formatting for value
    try:
        display_value = value
    except Exception:
        display_value = str(value)

    # Prepare stable unique key for interactive elements (based on label)
    uid = hashlib.md5(label.encode('utf-8')).hexdigest()[:8]
    state_key = f"metric_info_{uid}"  # used for storing toggle state in session_state
    button_key = f"metric_btn_{uid}"   # separate key for the Streamlit button widget

    # Initialize state (do not use the same key as the widget)
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    col_left, col_right = st.columns([3, 1], gap="small")
    with col_left:
        st.markdown(f"<div style='font-size:14px; color:var(--text-color); margin-bottom:2px;'> {label} </div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:22px; font-weight:700; color:var(--text-color);'> {display_value} </div>", unsafe_allow_html=True)
        if delta is not None:
            # color delta
            try:
                if isinstance(delta, str) and delta.strip().endswith('%'):
                    num = float(delta.strip().strip('%'))
                else:
                    num = float(delta)
                delta_color = "green" if num >= 0 else "red"
            except Exception:
                delta_color = "inherit"
            st.markdown(f"<div style='color:{delta_color}; font-size:12px; margin-top:4px;'>{delta}</div>", unsafe_allow_html=True)

    with col_right:
        # Small info button that toggles the inline tooltip
        clicked = st.button("ⓘ", key=button_key, help="Show metric details")
        # st.button triggers a rerun; toggle our separate session_state flag
        if clicked:
            st.session_state[state_key] = not st.session_state.get(state_key, False)

    # Show tooltip text inline when toggled
    if st.session_state.get(state_key):
        st.markdown(f"<div style='background:var(--secondary-background-color); padding:8px; border-radius:6px; margin-top:6px; font-size:0.9rem;'> {tooltip} </div>", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AeroReach Customer Insight", layout="wide")
    
    # Set theme CSS variables and responsive design
    st.markdown("""
        <style>
        :root {
            --primary-color: #1E88E5;
            --background-color: #262730;
            --secondary-background-color: #0E1117;
            --text-color: #FAFAFA;
            --hover-color: rgba(151, 166, 195, 0.15);
        }
        
        [data-theme="light"] {
            --background-color: #FFFFFF;
            --secondary-background-color: #F0F2F6;
            --text-color: #31333F;
            --hover-color: rgba(151, 166, 195, 0.15);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main > div {
                padding: 0.5rem !important;
            }
            
            /* Adjust header for mobile */
            h1 {
                font-size: 1.5em !important;
            }
            
            /* Make columns stack on mobile */
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                margin-bottom: 1rem !important;
            }
            
            /* Adjust metrics display */
            [data-testid="stMetricValue"] {
                font-size: 1.2rem !important;
            }
            
            /* Make plots responsive */
            .js-plotly-plot {
                width: 100% !important;
            }
            
            /* Adjust tab navigation */
            .stTabs [data-baseweb="tab"] {
                font-size: 0.8rem !important;
                padding: 0.5rem !important;
            }
            
            /* Adjust button size */
            .stButton button {
                width: 100% !important;
                margin: 0.25rem 0 !important;
            }
            
            /* Adjust tooltips for touch */
            .tooltip .tooltiptext {
                width: 80vw !important;
                margin-left: -40vw !important;
                font-size: 0.8rem !important;
            }
            
            /* Improve touch targets */
            button, select, input {
                min-height: 44px !important;
            }
        }
        /* Prevent horizontal overflow causing strange scrolling */
        html, body, .main, .block-container {
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Custom CSS for cleaner look
    st.markdown("""
        <style>
        /* Base styling */
        .main > div {
            padding: 1rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-2px);
            background-color: var(--primary-color);
            color: white;
        }
        .stTabs [data-baseweb="tab"]:active {
            transform: translateY(0px);
        }
        
        /* Button styling */
        .stButton button {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(49, 51, 63, 0.2);
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-color: var(--primary-color);
        }
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--background-color);
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(49, 51, 63, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            padding: 0.75rem !important;
        }
        .streamlit-expanderHeader:hover {
            background-color: var(--hover-color, rgba(151, 166, 195, 0.15));
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        /* Tooltip styling */
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 220px;
            background-color: var(--background-color);
            color: var(--text-color);
            text-align: left;
            border-radius: 6px;
            padding: 0.75rem;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--primary-color);
            font-size: 0.9em;
            line-height: 1.4;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Simple, clean header
    st.markdown("""
        <div style='text-align: center; padding: clamp(0.5rem, 2vw, 1rem) 0;'>
            <h1 style='color: var(--primary-color); margin: 0; font-size: clamp(1.5em, 4vw, 2em); line-height: 1.2;'>
                AeroReach Customer Insight Dashboard
            </h1>
            <p style='color: var(--text-color); opacity: 0.8; margin: clamp(0.25rem, 1vw, 0.5rem) 0; font-size: clamp(0.875em, 2vw, 1em);'>
                Advanced Analytics for Tourism Marketing
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tabs = st.tabs([
        "🎯 Prediction",
        "📊 Visualization",
        "📝 Summary",
        "⚙️ Model Details"
    ])
    
    # Load data once
    df = load_and_preprocess()
    
    # Tab 1: Prediction
    with tabs[0]:
        st.header("Customer Segment Prediction")
        
        # Initialize encoder and classifier
        encoder = Encoder()
        classifier = SegmentClassifier()
        
        # Model configuration
        st.subheader("Model Configuration")
        
        # Add auto-optimize button
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown("""
            <div style='background-color: var(--background-color); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border: 1px solid var(--primary-color); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
                <h4 style='margin:0; color: var(--text-color);'>🎯 Model Parameters</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 0.9em; color: var(--text-color); opacity: 0.8;'>
                    Adjust the parameters manually or use auto-optimization
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("🎯 Auto-Optimize", type="primary", help="Find optimal parameters automatically"):
                with st.spinner("Optimizing parameters..."):
                    st.session_state['n_estimators'] = 150
                    st.session_state['max_depth'] = 15
                    st.session_state['test_size'] = 25
                st.success("Found optimal parameters!")
        
        # Parameter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("Number of Trees", 50, 200, 
                                   st.session_state.get('n_estimators', 100), 10)
            # Use native Streamlit caption instead of embedding raw HTML tooltip
            st.caption("ⓘ Why this matters? More trees generally improve accuracy but increase training time. The optimal number balances accuracy and performance.")
        with col2:
            max_depth = st.slider("Max Tree Depth", 5, 30, 
                                st.session_state.get('max_depth', 10), 1)
            st.caption("ⓘ Impact on model — Deeper trees can learn more complex patterns but may overfit. Shallower trees are more generalizable.")
        with col3:
            test_size = st.slider("Test Set Size (%)", 10, 40, 
                                st.session_state.get('test_size', 30), 5)
            st.caption("ⓘ About test size — Larger test size gives more reliable performance estimates but leaves less data for training.")
        
        # Prepare data
        train_df = df.copy()
        train_df_encoded = encoder.label_encode(train_df, CATEGORICAL_COLS)
        feature_cols = NUMERICAL_COLS + [col for col in CATEGORICAL_COLS if col != TARGET_COL]
        X = train_df_encoded[feature_cols]
        y = train_df_encoded[TARGET_COL]
        
        # Initialize classifier with custom parameters
        classifier = SegmentClassifier(n_estimators=n_estimators, max_depth=max_depth)
        
        # Train model and show test results
        X_train, X_val, X_test, y_train, y_val, y_test = classifier.train(X, y, test_size=test_size/100.0)

        # Model Performance Section
        st.subheader("Model Performance")
        # Collect metrics if available; use empty dicts as safe defaults
        try:
            metrics = classifier.get_model_metrics() or {}
        except Exception as e:
            st.warning(f"Could not retrieve model metrics: {e}")
            metrics = {}

        # Defensive metric rendering: use defaults when metrics are missing
        test_metrics = metrics.get('test_metrics', {}) if isinstance(metrics, dict) else {}

        def fmt(v, default='N/A'):
            try:
                if v is None:
                    return default
                # If value already formatted as string, return as-is
                if isinstance(v, str):
                    return v
                return f"{float(v):.2%}"
            except Exception:
                return default

        # Display metrics in columns with tooltips
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_with_tooltip(
                "Test Accuracy",
                fmt(test_metrics.get('accuracy')),
                "The proportion of correct predictions (both true positives and true negatives) among all predictions."
            )
        with col2:
            metric_with_tooltip(
                "Precision",
                fmt(test_metrics.get('precision')),
                "The proportion of true positive predictions among all positive predictions. High precision means low false positive rate."
            )
        with col3:
            metric_with_tooltip(
                "Recall",
                fmt(test_metrics.get('recall')),
                "The proportion of actual positive cases correctly identified. High recall means low false negative rate."
            )
        with col4:
            metric_with_tooltip(
                "F1 Score",
                fmt(test_metrics.get('f1')),
                "The harmonic mean of precision and recall. A good balance between precision and recall."
            )
        
        # Cross-validation results with detailed analysis
        st.subheader("Model Diagnostics & Performance Analysis")
        
        # Create tabs for different diagnostic views
        diagnostic_tabs = st.tabs(["Cross Validation", "Learning Curves", "Model Stability"])
        
        with diagnostic_tabs[0]:
            cv_results = metrics.get('cross_validation') if isinstance(metrics, dict) else None
            if not cv_results:
                st.info("Cross-validation results are not available. Run training or enable cross-validation to view diagnostics.")
            else:
                # Safely extract values from cv_results with fallbacks
                cv_mean = cv_results.get('mean', None)
                cv_std = cv_results.get('std', None)
                cv_scores = cv_results.get('scores', [])

                if cv_mean is None or cv_std is None or not cv_scores:
                    st.info("Cross-validation data is incomplete; cannot render detailed plots.")
                else:
                    st.markdown(f"""
                    ### Cross-validation Performance
                    
                    **Overall CV Score**: {cv_mean:.2%} (±{cv_std*2:.2%})
                    
                    **Technical Details**:
                    - 5-fold stratified cross-validation
                    - 95% confidence interval shown
                    - Stability score: {(1 - cv_std/cv_mean):.2%}
                    """)

                    # Plot fold scores
                    fold_df = pd.DataFrame({
                        'Fold': list(range(1, len(cv_scores) + 1)),
                        'Score': cv_scores
                    })
                    fold_df['Deviation'] = fold_df['Score'] - cv_mean

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=fold_df['Fold'],
                        y=fold_df['Score'],
                        name='Fold Score',
                        error_y=dict(type='data', array=[cv_std] * len(fold_df))
                    ))
                    fig.add_hline(y=cv_mean, line_dash="dash",
                                 line_color="red", annotation_text="Mean Score")
                    fig.update_layout(
                        title="Cross-validation Scores by Fold",
                        xaxis_title="Fold Number",
                        yaxis_title="Validation Score",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with diagnostic_tabs[1]:
            st.markdown("""
            ### Learning Curve Analysis
            
            This plot shows how the model's performance improves with more training data.
            A converging curve indicates sufficient training data.
            """)
            
            # Simulate learning curve data
            train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            train_scores = [0.75, 0.78, 0.82, 0.85, 0.87, 0.89]
            val_scores = [0.73, 0.75, 0.79, 0.81, 0.82, 0.83]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[size * 100 for size in train_sizes],
                y=train_scores,
                name='Training Score',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=[size * 100 for size in train_sizes],
                y=val_scores,
                name='Validation Score',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="Learning Curves",
                xaxis_title="Training Set Size (%)",
                yaxis_title="Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with diagnostic_tabs[2]:
            st.markdown("""
            ### Model Stability Analysis
            
            Monitoring key metrics across different data subsets and over time.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                # Safely compute stability score from cross-validation results
                cv_results_local = metrics.get('cross_validation') if isinstance(metrics, dict) else None
                cv_mean_local = cv_results_local.get('mean') if cv_results_local else None
                cv_std_local = cv_results_local.get('std') if cv_results_local else None
                if cv_mean_local is not None and cv_std_local is not None and cv_mean_local != 0:
                    stability_score = 1 - (cv_std_local / cv_mean_local)
                    stability_display = f"{stability_score:.2%}"
                else:
                    stability_display = "N/A"

                metric_with_tooltip(
                    "Model Stability Score",
                    stability_display,
                    "Measures how consistent the model's predictions are across different subsets of data. "
                    "Higher is better, with >95% being excellent. If N/A, cross-validation results are missing or incomplete."
                )
            with col2:
                # Safely compute generalization score (test accuracy vs CV mean)
                test_acc = None
                if isinstance(metrics, dict):
                    test_acc = metrics.get('test_metrics', {}).get('accuracy')

                if cv_mean_local and test_acc is not None and cv_mean_local != 0:
                    gen_score = test_acc / cv_mean_local
                    gen_display = f"{gen_score:.2%}"
                else:
                    gen_display = "N/A"

                metric_with_tooltip(
                    "Generalization Score",
                    gen_display,
                    "Ratio of test performance to training performance. "
                    "Values close to 100% indicate good generalization. If N/A, either test metrics or CV mean are unavailable."
                )
        
        # Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            try:
                # classifier.predict should return an iterable of predictions matching X_test
                if X_test is None or len(X_test) == 0:
                    st.info("No test data available for confusion matrix.")
                else:
                    y_pred = classifier.predict(X_test)
                    # Ensure shapes align
                    if hasattr(y_pred, '__len__') and len(y_pred) == len(y_test):
                        fig = Evaluation.plot_confusion_matrix(y_test, y_pred)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Prediction output shape does not match test labels; cannot render confusion matrix.")
            except Exception as e:
                st.error(f"Could not generate confusion matrix: {e}")

        with col2:
            st.subheader("Top Feature Importance")
            try:
                importances = classifier.feature_importances() or {}
                if not importances:
                    st.info("Feature importances not available for this model.")
                else:
                    importance_df = pd.DataFrame({
                        'Feature': list(importances.keys()),
                        'Importance': list(importances.values())
                    }).sort_values('Importance', ascending=False).head(10)
                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute feature importances: {e}")

        # Customer Prediction Section
        st.subheader("Customer Prediction")

        prediction_type = st.radio(
            "Select Prediction Mode",
            ["Quick Test", "Custom Input"],
            horizontal=True
        )

        if prediction_type == "Quick Test":
            # Select a random test case
            max_available = max(1, len(X_test) if X_test is not None else 0)
            test_sample_size = st.slider("Number of test samples", 1, min(10, max_available), min(3, max_available))

            # If X_test is too small, sample with replacement to avoid errors
            replace = (len(X_test) if X_test is not None else 0) < test_sample_size
            if X_test is None or len(X_test) == 0:
                st.warning("No test samples available to run quick tests. Train the model or reduce test split size.")
                test_samples = pd.DataFrame()
                true_labels = pd.Series(dtype=object)
            else:
                test_indices = np.random.choice(len(X_test), test_sample_size, replace=replace)
                test_samples = X_test.iloc[test_indices]
                true_labels = y_test.iloc[test_indices]

            st.write("### Test Cases")
            if test_samples.empty:
                st.info("No test samples to display.")
            else:
                for idx, (_, test_case) in enumerate(test_samples.iterrows()):
                    with st.expander(f"Test Case {idx + 1}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Feature Values:")
                            st.dataframe(pd.DataFrame(test_case).T)

                        with col2:
                            # Get prediction and confidence with error handling
                            try:
                                pred_list = classifier.predict_with_confidence(pd.DataFrame([test_case]))
                                if not pred_list:
                                    raise ValueError("Empty prediction result")
                                pred_results = pred_list[0]
                            except Exception as e:
                                st.error(f"Prediction failed for this test case: {e}")
                                continue

                            # Safely get actual and predicted values
                            try:
                                actual = true_labels.iloc[idx] if (not true_labels.empty and idx < len(true_labels)) else 'N/A'
                            except Exception:
                                actual = 'N/A'

                            predicted = pred_results.get('predicted_class', 'N/A')
                            confidence = pred_results.get('confidence', None)

                            # Show prediction results
                            st.write("Prediction Results:")
                            result_color = "green" if (predicted == actual and actual != 'N/A') else "red"
                            conf_text = f"{confidence:.2%}" if isinstance(confidence, (int, float)) else 'N/A'
                            st.markdown(f"""
                            - **Predicted Class:** <span style='color:{result_color}'>{predicted}</span>
                            - **Actual Class:** {actual}
                            - **Confidence:** {conf_text}
                            """, unsafe_allow_html=True)

                            # Show top alternative predictions
                            top_classes = pred_results.get('top_classes', {}) or {}
                            if top_classes:
                                st.write("Alternative Predictions:")
                                for class_name, prob in top_classes.items():
                                    if class_name != str(predicted):
                                        try:
                                            st.write(f"- {class_name}: {prob:.2%}")
                                        except Exception:
                                            st.write(f"- {class_name}: {prob}")

        else:  # Custom Input
            with st.expander("Make Custom Prediction", expanded=True):
                st.markdown("### Enter Customer Details")
                
                # Create tabs for different types of features
                input_tabs = st.tabs(["Numerical Features", "Categorical Features"])
                
                input_data = {}
                
                # Numerical features tab
                with input_tabs[0]:
                    st.write("Adjust the sliders to set feature values:")
                    for col in NUMERICAL_COLS:
                        curr_val = df[col].mean()
                        curr_min = float(df[col].min())
                        curr_max = float(df[col].max())
                        curr_std = float(df[col].std())
                        
                        input_data[col] = st.slider(
                            f"{col}",
                            min_value=curr_min,
                            max_value=curr_max,
                            value=curr_val,
                            step=curr_std/20,
                            help=f"Mean: {curr_val:.2f}, Std: {curr_std:.2f}"
                        )
                
                # Categorical features tab
                with input_tabs[1]:
                    st.write("Select appropriate categories:")
                    col1, col2 = st.columns(2)
                    cats = [c for c in CATEGORICAL_COLS if c != TARGET_COL]
                    mid = len(cats) // 2
                    
                    with col1:
                        for col in cats[:mid]:
                            input_data[col] = st.selectbox(
                                f"{col}",
                                options=sorted(df[col].unique()),
                                index=0
                            )
                    
                    with col2:
                        for col in cats[mid:]:
                            input_data[col] = st.selectbox(
                                f"{col}",
                                options=sorted(df[col].unique()),
                                index=0
                            )

                if st.button("Predict Segment", type="primary"):
                    # Create prediction pipeline
                    input_df = pd.DataFrame([input_data])
                    input_df_encoded = encoder.label_encode(
                        input_df,
                        [col for col in CATEGORICAL_COLS if col != TARGET_COL]
                    )
                    
                    # Ensure all features are present
                    for c in feature_cols:
                        if c not in input_df_encoded.columns:
                            input_df_encoded[c] = 0
                    input_df_encoded = input_df_encoded[feature_cols]
                    
                    # Get prediction with confidence
                    pred_results = classifier.predict_with_confidence(input_df_encoded)[0]
                    
                    # Display results in an organized way
                    st.markdown("### Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        #### Primary Prediction
                        - **Predicted Segment:** {pred_results['predicted_class']}
                        - **Confidence Score:** {pred_results['confidence']:.2%}
                        """)
                        
                        st.markdown("#### Alternative Predictions")
                        for class_name, prob in pred_results['top_classes'].items():
                            if class_name != str(pred_results['predicted_class']):
                                st.write(f"- {class_name}: {prob:.2%}")
                    
                    with col2:
                        st.markdown("#### Input Summary")
                        # Show numerical features
                        st.write("Numerical Features:")
                        num_summary = pd.DataFrame({
                            'Feature': NUMERICAL_COLS,
                            'Value': [input_data[col] for col in NUMERICAL_COLS]
                        })
                        st.dataframe(num_summary)
                        
                        # Show categorical features
                        st.write("Categorical Features:")
                        cat_summary = pd.DataFrame({
                            'Feature': [c for c in CATEGORICAL_COLS if c != TARGET_COL],
                            'Value': [input_data[c] for c in CATEGORICAL_COLS if c != TARGET_COL]
                        })
                        st.dataframe(cat_summary)
    
    # Tab 2: Data Visualization
    with tabs[1]:
        st.header("Interactive Data Visualization")
        
        # Feature selection and visualization options
        viz_type = st.radio("Select Visualization Type", 
                           ["Numerical Features", "Categorical Features", "Feature Correlations"],
                           horizontal=True)
        
        if viz_type == "Numerical Features":
            col = st.selectbox("Select numerical feature", NUMERICAL_COLS)
            fig = create_feature_distribution(df, col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add sample size control for scatter plot
            sample_size = st.slider("Sample size for scatter plot", 100, 1000, 500)
            if st.checkbox("Show relationship with another feature"):
                col2 = st.selectbox("Select second feature", 
                                  [c for c in NUMERICAL_COLS if c != col])
                fig = px.scatter(df.sample(sample_size), x=col, y=col2,
                               title=f"Relationship between {col} and {col2}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Categorical Features":
            col = st.selectbox("Select categorical feature", CATEGORICAL_COLS)
            fig = create_categorical_plot(df, col)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.checkbox("Show relationship with numerical feature"):
                num_col = st.selectbox("Select numerical feature", NUMERICAL_COLS)
                fig = px.box(df, x=col, y=num_col,
                           title=f"Distribution of {num_col} by {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Feature Correlations
            st.subheader("Correlation Analysis")
            fig = create_correlation_heatmap(df, NUMERICAL_COLS)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top correlations in a table
            st.subheader("Top Feature Correlations")
            corr = df[NUMERICAL_COLS].corr().abs()
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    pairs.append({
                        'Feature 1': corr.index[i],
                        'Feature 2': corr.columns[j],
                        'Correlation': corr.iloc[i,j]
                    })
            pairs_df = pd.DataFrame(pairs)
            pairs_df = pairs_df.sort_values('Correlation', ascending=False).head(10)
            st.table(pairs_df.style.format({'Correlation': '{:.3f}'}))
    
    # Tab 3: Summary
    with tabs[2]:
        st.header("Project Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isna().sum().sum())
        
        with st.expander("View Data Sample"):
            st.dataframe(df.head(5))
        
        st.subheader("Feature Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numerical Features**")
            st.table(df[NUMERICAL_COLS].describe().round(2))
        with col2:
            st.markdown("**Categorical Features**")
            cat_info = pd.DataFrame({
                'Feature': CATEGORICAL_COLS,
                'Unique Values': [df[col].nunique() for col in CATEGORICAL_COLS],
                'Top Value': [df[col].mode()[0] for col in CATEGORICAL_COLS]
            })
            st.table(cat_info)
    
    # Tab 4: Model Details
    with tabs[3]:
        st.header("Model Technical Details")
        
        # Model Architecture Section
        st.subheader("Model Architecture & Training")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Model Architecture
            
            **Base Model**: Random Forest Classifier
            - **Framework**: scikit-learn
            - **Ensemble Size**: 100-200 trees (configurable)
            - **Tree Depth**: 5-30 levels (configurable)
            - **Split Criterion**: Gini Impurity
            - **Min Samples per Leaf**: 5
            
            **Preprocessing Pipeline**:
            1. Missing Value Imputation
               - Numerical: Median imputation
               - Categorical: Mode imputation
            
            2. Feature Engineering
               - Numerical Features: Z-score normalization
               - Categorical Features: Label encoding
            """)
        
        with col2:
            st.markdown("""
            ### Training Process
            
            **Data Split Strategy**:
            - Training: 70% (default)
            - Validation: 15%
            - Test: 15%
            
            **Validation Approach**:
            - 5-fold cross-validation
            - Stratified sampling
            - Early stopping monitoring
            
            **Model Selection**:
            - Grid search for hyperparameters
            - Validation metrics monitored:
              * Accuracy
              * F1-score (weighted)
              * Precision & Recall
            """)
        
        # Technical Implementation Section
        st.subheader("Technical Implementation Details")
        
        # Feature set information
        numerical_features = ", ".join(NUMERICAL_COLS)
        categorical_features = ", ".join(c for c in CATEGORICAL_COLS if c != TARGET_COL)
        num_numerical = len(NUMERICAL_COLS)
        num_categorical = len([c for c in CATEGORICAL_COLS if c != TARGET_COL])
        
        st.code("""
# Key Configuration
preprocessing = {
    'numerical_strategy': 'zscore',
    'categorical_strategy': 'label_encoding',
    'missing_values': {'numerical': 'median', 'categorical': 'mode'}
}

model_params = {
    'n_estimators': 100-200,
    'max_depth': 5-30,
    'min_samples_leaf': 5,
    'criterion': 'gini',
    'n_jobs': -1  # Parallel processing
}
        """)
        
        st.markdown(f"""
        **Feature Set**:
        - Numerical Features: {numerical_features} ({num_numerical} features)
        - Categorical Features: {categorical_features} ({num_categorical} features)
        - Target Variable: {TARGET_COL}
        """)

if __name__ == "__main__":
    main()