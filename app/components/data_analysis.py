import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.data_utils import (
    load_dataset, 
    preprocess_data, 
    generate_correlation_heatmap,
    generate_scatter_plot,
    generate_feature_importance_plot
)
from sklearn.preprocessing import LabelEncoder

def run_data_analysis():
    """Main function for the data analysis tab"""
    st.markdown("## 📊 Data Analysis & Visualization")
    st.markdown("""
    Explore the student performance dataset through interactive visualizations. 
    Understand the relationships between different features and their impact on performance.
    """)
    
    # Load the dataset
    df = load_dataset()
    if df is None:
        st.error("Failed to load dataset. Please check the file path.")
        return
    
    # Display dataset overview
    with st.expander("Dataset Overview", expanded=True):
        show_dataset_overview(df)
    
    # Statistical analysis
    with st.expander("Statistical Analysis", expanded=False):
        show_statistical_analysis(df)
    
    # Feature correlation
    with st.expander("Feature Correlation Analysis", expanded=True):
        show_correlation_analysis(df)
    
    # Feature visualization
    with st.expander("Feature Visualization", expanded=True):
        show_feature_visualization(df)
    
    # Performance distribution
    with st.expander("Performance Distribution", expanded=True):
        show_performance_distribution(df)

def show_dataset_overview(df):
    """Display dataset overview"""
    st.markdown("### Dataset Overview")
    
    # Dataset info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Records", len(df))
        st.metric("Number of Features", len(df.columns) - 1)  # Excluding target variable
    
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Target Variable", "Performance Index")
    
    # Display the first few rows
    st.markdown("#### Sample Data")
    st.dataframe(df.head(5), use_container_width=True)
    
    # Column information
    st.markdown("#### Column Information")
    column_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(column_info, use_container_width=True)

def show_statistical_analysis(df):
    """Display statistical analysis"""
    st.markdown("### Statistical Analysis")
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Descriptive statistics for numeric columns
    st.markdown("#### Descriptive Statistics (Numeric Columns)")
    st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.markdown("#### Categorical Columns Summary")
        for col in categorical_cols:
            st.markdown(f"**{col}**")
            st.dataframe(df[col].value_counts().reset_index().rename(
                columns={"index": col, col: "Count"}), use_container_width=True)
    
    # Target variable distribution
    st.markdown("#### Target Variable Distribution")
    
    # Create histogram with Plotly
    fig = px.histogram(
        df, x="Performance Index", 
        nbins=20,
        color_discrete_sequence=['#1e3a8a'],
        opacity=0.7,
        marginal="box"
    )
    
    fig.update_layout(
        title="Distribution of Performance Index",
        xaxis_title="Performance Index",
        yaxis_title="Count",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Skewness and kurtosis
    col1, col2 = st.columns(2)
    with col1:
        skewness = df["Performance Index"].skew()
        st.metric("Skewness", f"{skewness:.2f}", 
                 delta=None if abs(skewness) < 0.5 else "Indicates skewed distribution")
    
    with col2:
        kurtosis = df["Performance Index"].kurtosis()
        st.metric("Kurtosis", f"{kurtosis:.2f}", 
                 delta=None if abs(kurtosis) < 0.5 else "Non-normal distribution")

def show_correlation_analysis(df):
    """Display correlation analysis"""
    st.markdown("### Correlation Analysis")
    st.markdown("""
    This heatmap shows the correlation between different features. 
    Strong positive correlations are shown in dark blue, and strong negative correlations in dark red.
    """)
    
    # Create correlation heatmap
    fig = generate_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights about correlations
    st.markdown("#### Key Insights")
    
    # Convert categorical columns to numeric for correlation calculation
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            # For binary categorical variables like "Yes"/"No"
            if set(df_numeric[col].unique()).issubset({"Yes", "No"}):
                df_numeric[col] = df_numeric[col].map({"Yes": 1, "No": 0})
            else:
                # For other categorical variables, use label encoding
                le = LabelEncoder()
                df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
    
    # Calculate correlations with target
    corr_with_target = df_numeric.corr()["Performance Index"].sort_values(ascending=False).drop("Performance Index")
    
    # Display top positive and negative correlations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Top Positive Correlations")
        top_positive = corr_with_target[corr_with_target > 0].head(3)
        for feature, corr in top_positive.items():
            st.markdown(f"**{feature}**: {corr:.2f}")
    
    with col2:
        st.markdown("##### Top Negative Correlations")
        top_negative = corr_with_target[corr_with_target < 0].head(3)
        for feature, corr in top_negative.items():
            st.markdown(f"**{feature}**: {corr:.2f}")
    
    # Correlation interpretation
    st.markdown("#### Correlation Interpretation")
    st.markdown("""
    - A correlation coefficient of 1 indicates a perfect positive correlation
    - A correlation coefficient of -1 indicates a perfect negative correlation
    - A correlation coefficient of 0 indicates no linear correlation
    """)

def show_feature_visualization(df):
    """Display feature visualizations"""
    st.markdown("### Feature Visualization")
    st.markdown("Explore the relationship between different features and the Performance Index.")
    
    # Feature selector
    features = [col for col in df.columns if col != "Performance Index"]
    selected_feature = st.selectbox("Select Feature to Visualize", features)
    
    # Create visualizations based on feature type
    if df[selected_feature].dtype in ["int64", "float64"]:
        show_numeric_feature_visualization(df, selected_feature)
    else:
        show_categorical_feature_visualization(df, selected_feature)
    
    # Two-feature relationship
    st.markdown("#### Relationship Between Two Features")
    
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("Select X-axis Feature", features, index=0)
    with col2:
        feature_y = st.selectbox("Select Y-axis Feature", features, index=1 if len(features) > 1 else 0)
    
    # Create scatter plot
    fig = generate_scatter_plot(df, feature_x, feature_y, "Performance Index")
    st.plotly_chart(fig, use_container_width=True)

def show_numeric_feature_visualization(df, feature):
    """Display visualizations for numeric features"""
    col1, col2 = st.columns(2)
    
    # Scatter plot
    with col1:
        try:
            # Try with trendline
            fig_scatter = px.scatter(
                df, x=feature, y="Performance Index",
                color="Performance Index",
                color_continuous_scale="viridis",
                opacity=0.7,
                trendline="ols"
            )
        except ImportError:
            # Fall back to no trendline if statsmodels is not available
            fig_scatter = px.scatter(
                df, x=feature, y="Performance Index",
                color="Performance Index",
                color_continuous_scale="viridis",
                opacity=0.7
            )
            st.info("Install statsmodels package for trend line visualization.")
        
        fig_scatter.update_layout(
            title=f"{feature} vs Performance Index",
            xaxis_title=feature,
            yaxis_title="Performance Index",
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Histogram/Distribution
    with col2:
        fig_hist = px.histogram(
            df, x=feature,
            color_discrete_sequence=['#4c78a8'],
            opacity=0.7,
            nbins=20
        )
        
        fig_hist.update_layout(
            title=f"Distribution of {feature}",
            xaxis_title=feature,
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Box plot
    fig_box = px.box(
        df, y=feature,
        color_discrete_sequence=['#4c78a8']
    )
    
    fig_box.update_layout(
        title=f"Box Plot of {feature}",
        yaxis_title=feature,
        height=300
    )
    
    st.plotly_chart(fig_box, use_container_width=True)

def show_categorical_feature_visualization(df, feature):
    """Display visualizations for categorical features"""
    # Bar chart of average performance by category
    avg_performance = df.groupby(feature)["Performance Index"].mean().reset_index()
    
    fig_bar = px.bar(
        avg_performance, 
        x=feature, 
        y="Performance Index",
        color="Performance Index",
        color_continuous_scale="viridis",
        title=f"Average Performance by {feature}"
    )
    
    fig_bar.update_layout(
        xaxis_title=feature,
        yaxis_title="Average Performance Index",
        height=400
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Count plot
    fig_count = px.histogram(
        df, x=feature,
        color=feature,
        title=f"Count of Students by {feature}"
    )
    
    fig_count.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
        height=400
    )
    
    st.plotly_chart(fig_count, use_container_width=True)
    
    # Box plot of performance by category
    fig_box = px.box(
        df, x=feature, y="Performance Index",
        color=feature,
        title=f"Performance Distribution by {feature}"
    )
    
    fig_box.update_layout(
        xaxis_title=feature,
        yaxis_title="Performance Index",
        height=400
    )
    
    st.plotly_chart(fig_box, use_container_width=True)

def show_performance_distribution(df):
    """Display performance distribution analysis"""
    st.markdown("### Performance Distribution Analysis")
    
    # Create performance categories for analysis
    df_copy = df.copy()
    performance_bins = [0, 40, 60, 75, 90, 100]
    performance_labels = ['Poor', 'Average', 'Good', 'Very Good', 'Excellent']
    df_copy['Performance Category'] = pd.cut(df_copy['Performance Index'], bins=performance_bins, labels=performance_labels)
    
    # Performance category distribution
    fig_pie = px.pie(
        df_copy, 
        names='Performance Category',
        color='Performance Category',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Distribution of Performance Categories"
    )
    
    fig_pie.update_layout(height=500)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Performance by different features
    st.markdown("#### Performance by Feature")
    
    feature_for_comparison = st.selectbox(
        "Select Feature for Comparison",
        [col for col in df.columns if col not in ["Performance Index", "Performance Category"]]
    )
    
    # Show appropriate visualization based on feature type
    if df[feature_for_comparison].dtype in ["int64", "float64"]:
        # For numeric features, show average value by performance category
        avg_by_category = df_copy.groupby('Performance Category')[feature_for_comparison].mean().reset_index()
        
        fig_bar = px.bar(
            avg_by_category,
            x='Performance Category',
            y=feature_for_comparison,
            color='Performance Category',
            color_discrete_sequence=px.colors.sequential.Viridis,
            title=f"Average {feature_for_comparison} by Performance Category"
        )
        
        fig_bar.update_layout(height=500)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        # For categorical features, show count by performance category
        category_counts = df_copy.groupby(['Performance Category', feature_for_comparison]).size().reset_index(name='Count')
        
        fig_group = px.bar(
            category_counts,
            x='Performance Category',
            y='Count',
            color=feature_for_comparison,
            barmode='group',
            title=f"Performance Categories by {feature_for_comparison}"
        )
        
        fig_group.update_layout(height=500)
        st.plotly_chart(fig_group, use_container_width=True) 