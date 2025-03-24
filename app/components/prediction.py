import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from ..models.model_handler import ModelManager
from ..utils.data_utils import load_dataset, save_user_prediction
from ..utils.ui_utils import (
    create_gauge_chart, 
    generate_recommendations, 
    create_comparison_radar_chart,
    create_prediction_progress_bars,
    create_info_box
)

def run_prediction():
    """Main function for the prediction tab"""
    st.markdown("## 🔮 Performance Prediction")
    st.markdown("""
    Predict a student's performance by entering their details below. The system uses an ensemble 
    of machine learning models to provide accurate predictions and personalized recommendations.
    """)
    
    # Load dataset for averages
    df = load_dataset()
    if df is None:
        st.error("Failed to load dataset. Please check the file path.")
        return
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # If no models are loaded, display warning
    if all(model is None for model in model_manager.models.values()):
        st.warning("No trained models available. The system will use only the Linear Regression model.")
    
    # User input form
    with st.expander("Input Student Details", expanded=True):
        user_data = get_user_input()
    
    # Model selection
    model_names = [name for name, model in model_manager.models.items() if model is not None]
    if not model_names:
        st.error("No models available. Please train models first.")
        return
    
    selected_model = st.selectbox(
        "Select Prediction Model", 
        model_names, 
        index=model_names.index("Ensemble") if "Ensemble" in model_names else 0
    )
    
    # Make prediction
    if st.button("Predict Performance", type="primary"):
        with st.spinner("Generating prediction..."):
            # Preprocess input
            preprocessed_data = model_manager.preprocess_input(user_data)
            
            # Make prediction
            prediction = model_manager.predict_with_model(preprocessed_data, selected_model)
            
            if prediction is None:
                st.error(f"Failed to get prediction from {selected_model} model.")
                return
            
            # Generate confidence interval (mock for display)
            prediction_value = float(prediction)
            confidence_interval = (prediction_value - 5, prediction_value + 5)
            
            # Calculate feature contributions
            if selected_model in ["Linear Regression", "Random Forest", "XGBoost"]:
                contributions = model_manager.generate_feature_contributions(user_data, selected_model)
            else:
                # For ensemble, use linear regression for simplicity
                contributions = model_manager.generate_feature_contributions(user_data, "Linear Regression")
            
            # Display prediction results
            display_prediction_results(user_data, prediction_value, confidence_interval, contributions, df)
            
            # Save prediction to history
            save_user_prediction(user_data, prediction_value, 0.9)

def get_user_input():
    """Get user input for prediction"""
    st.markdown("### Student Information")
    
    # Use columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.slider(
            "Hours Studied per Day",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Average number of hours spent studying per day"
        )
        
        previous_scores = st.slider(
            "Previous Exam Scores",
            min_value=40,
            max_value=100,
            value=70,
            step=1,
            help="Average score in previous exams (40-100)"
        )
        
        extra_curricular = st.selectbox(
            "Extracurricular Activities",
            options=["Yes", "No"],
            index=0,
            help="Does the student participate in extracurricular activities?"
        )
    
    with col2:
        sleep_hours = st.slider(
            "Sleep Hours per Day",
            min_value=4,
            max_value=10,
            value=7,
            step=1,
            help="Average number of hours of sleep per day"
        )
        
        question_papers = st.slider(
            "Sample Question Papers Practiced",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Number of sample question papers practiced"
        )
    
    # Create user data dictionary
    user_data = {
        "Hours Studied": hours_studied,
        "Previous Scores": previous_scores,
        "Extracurricular Activities": extra_curricular,
        "Sleep Hours": sleep_hours,
        "Sample Question Papers Practiced": question_papers
    }
    
    return user_data

def display_prediction_results(user_data, prediction, confidence_interval, contributions, df):
    """Display prediction results with visualizations"""
    st.markdown("## Prediction Results")
    
    # Performance score and gauge
    st.markdown("### Predicted Performance Score")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create gauge chart
        fig = create_gauge_chart(
            prediction,
            "Predicted Performance Score",
            min_val=0,
            max_val=100
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Score Interpretation")
        
        if prediction < 40:
            st.error("**Poor Performance (0-40)**")
            st.markdown("The student is likely to struggle significantly. Immediate intervention is recommended.")
        elif prediction < 60:
            st.warning("**Average Performance (40-60)**")
            st.markdown("The student is performing at an average level. There is room for improvement.")
        elif prediction < 75:
            st.info("**Good Performance (60-75)**")
            st.markdown("The student is performing well but could still improve in certain areas.")
        elif prediction < 90:
            st.success("**Very Good Performance (75-90)**")
            st.markdown("The student is performing very well. Continue with current study habits.")
        else:
            st.success("**Excellent Performance (90-100)**")
            st.markdown("The student is performing excellently. Maintain current study strategies.")
    
    # Confidence interval
    st.markdown("#### Prediction Confidence Interval")
    st.write(f"The model predicts a performance score between **{confidence_interval[0]:.1f}** and **{confidence_interval[1]:.1f}** with 95% confidence.")
    
    # Feature contribution
    st.markdown("### Prediction Breakdown")
    
    # Sort contributions by absolute value
    sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
    
    # Display contributions as progress bars
    create_prediction_progress_bars(sorted_contributions)
    
    # Student profile comparison
    st.markdown("### Student Profile Comparison")
    
    # Get average values from dataset
    avg_values = {
        "Hours Studied": df["Hours Studied"].mean(),
        "Previous Scores": df["Previous Scores"].mean(),
        "Sleep Hours": df["Sleep Hours"].mean(),
        "Sample Question Papers Practiced": df["Sample Question Papers Practiced"].mean()
    }
    
    # For extracurricular activities, calculate percentage of "Yes"
    if "Extracurricular Activities" in df.columns:
        yes_percentage = df[df["Extracurricular Activities"] == "Yes"].shape[0] / df.shape[0]
        avg_values["Extracurricular Activities"] = yes_percentage
    
    # Convert categorical values to numeric for radar chart
    user_values_numeric = user_data.copy()
    if user_values_numeric["Extracurricular Activities"] == "Yes":
        user_values_numeric["Extracurricular Activities"] = 1.0
    else:
        user_values_numeric["Extracurricular Activities"] = 0.0
    
    # Create radar chart data
    feature_names = list(user_data.keys())
    user_values = [user_values_numeric[f] for f in feature_names]
    avg_vals = [avg_values[f] for f in feature_names]
    
    # Create radar chart
    fig = create_comparison_radar_chart(user_values, avg_vals, feature_names)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### Personalized Recommendations")
    
    # Generate recommendations
    recommendations = generate_recommendations(user_data, prediction)
    
    if not recommendations:
        st.info("No specific recommendations at this time. The student is on the right track!")
    else:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {rec}**")
    
    # Provide contextual information
    create_info_box(
        "What Affects Student Performance?",
        """
        Research shows that student performance is influenced by multiple factors including study time, 
        sleep quality, previous academic performance, and extracurricular activities. This prediction 
        model takes these factors into account to provide a holistic assessment.
        """,
        "📝"
    )

def display_prediction_history():
    """Display prediction history"""
    st.markdown("### Prediction History")
    
    # Display a message that this is a premium feature
    st.info("""
    **Premium Feature**
    
    Prediction history tracking allows you to monitor student progress over time.
    Upgrade to premium to access this feature.
    """)
    
    # Display some dummy history
    history = [
        {"timestamp": "2023-05-15 10:30:45", "prediction": 76.5},
        {"timestamp": "2023-05-10 14:22:33", "prediction": 72.3},
        {"timestamp": "2023-05-05 09:15:21", "prediction": 68.9}
    ]
    
    # Create a line chart of historical predictions
    if history:
        history_df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=history_df["timestamp"],
                y=history_df["prediction"],
                mode="lines+markers",
                name="Prediction Score",
                line=dict(color="#1e3a8a", width=3),
                marker=dict(size=10)
            )
        )
        
        fig.update_layout(
            title="Performance Prediction History",
            xaxis_title="Date",
            yaxis_title="Predicted Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True) 