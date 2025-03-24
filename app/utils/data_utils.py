import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
import os
import json

def load_dataset(file_name="Student-Performance.csv"):
    """
    Load and preprocess the student performance dataset
    
    Parameters:
    -----------
    file_name : str
        Name of the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    try:
        # First try in the app/data directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        file_path = os.path.join(data_dir, file_name)
        
        # If file not found in app/data, try in the root directory
        if not os.path.exists(file_path):
            project_root = os.path.dirname(base_dir)
            file_path = os.path.join(project_root, file_name)
            
            # Create data directory if it doesn't exist
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
        
        # Print debug info
        print(f"Loading dataset from: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Print dataframe info for debugging
        print("DataFrame info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types: {df.dtypes}")
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for analysis and modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    tuple
        (preprocessed DataFrame, X features, y target, feature_names)
    """
    # Make a copy to avoid modifying the original data
    data = df.copy()
    
    # Handle missing values if any
    if data.isnull().sum().sum() > 0:
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            data[col].fillna(data[col].median(), inplace=True)
            
        # Fill categorical columns with mode
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col])
    
    # Define features and target
    X = data.drop('Performance Index', axis=1)
    y = data['Performance Index']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_names = X.columns.tolist()
    
    return data, X_scaled, y, feature_names, scaler, le

def generate_correlation_heatmap(df):
    """
    Generate correlation heatmap using plotly
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive correlation heatmap
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_numeric = df.copy()
    
    # Convert categorical columns to numeric
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            # For binary categorical variables like "Yes"/"No"
            if set(df_numeric[col].unique()).issubset({"Yes", "No"}):
                df_numeric[col] = df_numeric[col].map({"Yes": 1, "No": 0})
            else:
                # For other categorical variables, use label encoding
                le = LabelEncoder()
                df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
    
    # Calculate correlation matrix
    corr = df_numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            z=corr_masked,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation')
        )
    )
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=600,
        width=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    return fig

def generate_feature_importance_plot(importance, feature_names):
    """
    Create a bar chart of feature importance
    
    Parameters:
    -----------
    importance : array-like
        Feature importance values
    feature_names : list
        List of feature names
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive feature importance chart
    """
    # Sort features by importance
    indices = np.argsort(importance)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            y=[feature_names[i] for i in indices],
            x=[importance[i] for i in indices],
            orientation='h',
            marker=dict(
                color=[importance[i] for i in indices],
                colorscale='viridis'
            )
        )
    )
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=500,
        width=700
    )
    
    return fig

def generate_scatter_plot(df, x_col, y_col, color_col=None):
    """
    Generate scatter plot with optional coloring
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to visualize
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    color_col : str, optional
        Column name for color coding
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive scatter plot
    """
    if color_col:
        fig = px.scatter(
            df, x=x_col, y=y_col, 
            color=color_col,
            title=f'{y_col} vs {x_col} by {color_col}',
            labels={x_col: x_col, y_col: y_col},
            opacity=0.7,
            color_continuous_scale='viridis'
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            title=f'{y_col} vs {x_col}',
            labels={x_col: x_col, y_col: y_col},
            opacity=0.7
        )
        
    fig.update_layout(
        height=500,
        width=700,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

def save_user_prediction(user_data, prediction, confidence=None):
    """
    Save user prediction to history
    
    Parameters:
    -----------
    user_data : dict
        User input data
    prediction : float
        Predicted score
    confidence : float, optional
        Prediction confidence
    """
    history_file = "app/data/prediction_history.json"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new prediction
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = {
        "timestamp": timestamp,
        "input_data": user_data,
        "prediction": float(prediction)
    }
    
    if confidence is not None:
        entry["confidence"] = float(confidence)
    
    history.append(entry)
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=4) 