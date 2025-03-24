import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import time

# Add the app directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from app.components.data_analysis import run_data_analysis
from app.components.model_training import run_model_training
from app.components.prediction import run_prediction, display_prediction_history
from app.utils.ui_utils import (
    set_page_config, 
    create_header, 
    create_sidebar_header,
    display_lottie_animation
)

# Set up the app configuration
set_page_config()

# Create the header
create_header()

# Create sidebar header
create_sidebar_header()

# Create sidebar menu
with st.sidebar:
    st.markdown("---")
    selected_tab = option_menu(
        "Navigation",
        ["Home", "Data Analysis", "Model Training", "Prediction", "About"],
        icons=["house", "bar-chart", "gear", "magic", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "#1e3a8a", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#1e3a8a", "color": "white"},
        }
    )
    
    # Display version info
    st.markdown("---")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Last Updated:** 2023-05-20")
    
    # Add a GitHub link
    st.markdown("---")
    st.markdown("[View Source Code on GitHub](https://github.com/shivsharankumar/student-performance-analytics)")
    
    # Add a signature
    st.markdown("---")
    st.markdown("Made with ❤️ by Shiv Sharan")

# Main content area
if selected_tab == "Home":
    # Display welcome message and app overview
    st.markdown("## 👋 Welcome to Student Performance Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This advanced analytics platform helps educators and students predict academic performance 
        using state-of-the-art machine learning techniques. The application integrates multiple 
        models and provides detailed analyses to help improve student outcomes.
        
        ### 🌟 Key Features
        
        - **Interactive Data Visualization**: Explore the relationships between different factors affecting student performance
        - **Advanced ML Models**: Utilize ensemble methods combining Linear Regression, Random Forest, and XGBoost
        - **Detailed Performance Breakdown**: Understand how each factor contributes to the predicted performance
        - **Personalized Recommendations**: Get tailored suggestions for improvement based on individual profiles
        - **Model Comparison**: Compare different machine learning models and their prediction accuracy
        
        ### 🚀 Getting Started
        
        1. Explore the **Data Analysis** tab to understand the patterns in student performance data
        2. Check out the **Model Training** tab to train and evaluate different prediction models
        3. Use the **Prediction** tab to get personalized performance predictions and recommendations
        
        Select a section from the sidebar to begin!
        """)
    
    with col2:
        # Add animation for visual appeal
        lottie_url = "https://assets9.lottiefiles.com/packages/lf20_kkflmtur.json"  # Education/data analytics animation
        display_lottie_animation(lottie_url, height=300)
    
    # Display feature highlights
    st.markdown("## ✨ Feature Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Interactive Analytics
        Explore student data through interactive visualizations that help identify key performance factors.
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Ensemble Prediction
        Utilize multiple machine learning models combined through a weighted ensemble approach for higher accuracy.
        """)
    
    with col3:
        st.markdown("""
        ### 🧠 Smart Recommendations
        Receive AI-generated, personalized recommendations to improve academic performance.
        """)
    
    # Add a call to action
    st.markdown("---")
    st.markdown("## Ready to Get Started?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Explore Data Analysis", type="primary"):
            st.session_state.selected_tab = "Data Analysis"
            st.rerun()
    
    with col2:
        if st.button("Train Models"):
            st.session_state.selected_tab = "Model Training"
            st.rerun()
    
    with col3:
        if st.button("Make Predictions"):
            st.session_state.selected_tab = "Prediction"
            st.rerun()

elif selected_tab == "Data Analysis":
    # Run the data analysis component
    run_data_analysis()

elif selected_tab == "Model Training":
    # Run the model training component
    run_model_training()

elif selected_tab == "Prediction":
    # Run the prediction component
    run_prediction()
    
    # Display prediction history
    with st.expander("Prediction History", expanded=False):
        display_prediction_history()

elif selected_tab == "About":
    # Display about information
    st.markdown("## About Student Performance Analytics")
    
    st.markdown("""
    ### Project Overview
    
    Student Performance Analytics is an advanced machine learning platform designed to help educators 
    and students predict and improve academic performance. The platform utilizes multiple machine 
    learning models, interactive data visualization, and personalized recommendations to provide 
    a comprehensive solution for educational analytics.
    
    ### Technical Details
    
    - **Frontend**: Streamlit, HTML, CSS
    - **Backend**: Python
    - **ML Libraries**: scikit-learn, XGBoost, SHAP
    - **Data Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    
    ### Data Sources
    
    The dataset used in this application includes student performance metrics collected from 
    various educational institutions. The data has been anonymized to protect student privacy.
    
    ### About the Creator
    
    This application was developed by [Your Name], a data scientist and educator passionate about 
    using machine learning to improve educational outcomes. For more information, please visit 
    [your website/LinkedIn profile].
    
    ### Acknowledgements
    
    Special thanks to all the educators and students who contributed to this project by providing 
    feedback and suggestions for improvement.
    
    ### Contact
    
    For questions, feedback, or collaboration opportunities, please contact:
    - Email: shivsharan47@gmail.com
    - GitHub: @shivsharan47
    - LinkedIn: https://www.linkedin.com/in/shiv-sharan-kumar-93aa3219b
    """)

# Add a footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    © 2023 Student Performance Analytics | All Rights Reserved
</div>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    # This block is executed when the script is run directly
    pass 