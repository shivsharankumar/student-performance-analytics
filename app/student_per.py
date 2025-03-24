import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import pickle
from sklearn.preprocessing import StandardScaler

def load_model():
    # Get the directory where the current file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the app directory
    app_dir = os.path.dirname(current_dir)
    # Construct the path to the model file
    model_path = os.path.join(app_dir, 'linear_regression_model.pkl')
    
    try:
        with open(model_path, 'rb') as file:
            model, scaler, le = pickle.load(file)
        return model, scaler, le
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        st.info("Please train the model first using the Model Training tab.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_data(data,scaler,le):
    # Transform categorical data
    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']])[0]
    
    # Create DataFrame with correct feature names that match the training data
    df=pd.DataFrame([data])
    
    # Ensure column names match exactly what was used during training
    # Rename columns to match training data if needed
    if 'Question Paper' in df.columns:
        df.rename(columns={'Question Paper': 'Sample Question Papers Practiced'}, inplace=True)
    if 'previous scores' in df.columns:
        df.rename(columns={'previous scores': 'Previous Scores'}, inplace=True)
    
    # Apply the scaling transformation
    df_transformed=scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model()
    if model is None or scaler is None or le is None:
        return None
    preprocessed_data = preprocess_data(data, scaler, le)
    predictions = model.predict(preprocessed_data)
    return predictions[0]


def main():
    """
    The main function sets up a simple interface for predicting student performance by prompting the
    user to enter the number of hours studied.
    """
    st.title("Student Performance Prediction")
    st.write("Enter the following details to predict the student's performance:")
    hours_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
    previous_scores = st.number_input("Previous Scores", min_value=40, max_value=100, value=70)
    extra_curricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    sleep_hours = st.number_input("Sleep Hours", min_value=4, max_value=10, value=7)
    question_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=10, value=5)
    if st.button("Predict"):
        user_data = {
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extra_curricular,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": question_papers
        }
        prediction = predict_data(user_data)
        if prediction is not None:
            st.success(f"The predicted score is {prediction:.2f}")


if __name__ == "__main__":
    main()

