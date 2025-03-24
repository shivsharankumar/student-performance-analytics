import pickle
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

class ModelManager:
    """Class to manage multiple models for student performance prediction"""
    
    def __init__(self, model_dir="app/models"):
        """Initialize the model manager"""
        self.model_dir = model_dir
        self.models = {
            "Linear Regression": None,
            "Random Forest": None,
            "XGBoost": None,
            "Ensemble": None
        }
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.model_metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Try to load existing models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            # Load the base linear regression model that was already created
            with open('linear_regression_model.pkl', 'rb') as file:
                linear_model, self.scaler, self.label_encoder = pickle.load(file)
                self.models["Linear Regression"] = linear_model
                
            # Load other models if they exist
            model_paths = {
                "Random Forest": os.path.join(self.model_dir, "random_forest_model.pkl"),
                "XGBoost": os.path.join(self.model_dir, "xgboost_model.pkl"),
                "Ensemble": os.path.join(self.model_dir, "ensemble_model.pkl")
            }
            
            for model_name, model_path in model_paths.items():
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    
            # Load feature names
            feature_path = os.path.join(self.model_dir, "feature_names.pkl")
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    self.feature_names = pickle.load(f)
                    
            # Load model metrics
            metrics_path = os.path.join(self.model_dir, "model_metrics.pkl")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'rb') as f:
                    self.model_metrics = pickle.load(f)
                    
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            # Save individual models
            for model_name, model in self.models.items():
                if model is not None:
                    model_path = os.path.join(self.model_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
                    joblib.dump(model, model_path)
            
            # Save feature names
            if self.feature_names is not None:
                feature_path = os.path.join(self.model_dir, "feature_names.pkl")
                with open(feature_path, 'wb') as f:
                    pickle.dump(self.feature_names, f)
            
            # Save model metrics
            metrics_path = os.path.join(self.model_dir, "model_metrics.pkl")
            with open(metrics_path, 'wb') as f:
                pickle.dump(self.model_metrics, f)
                
            return True
                
        except Exception as e:
            st.error(f"Error saving models: {e}")
            return False
    
    def train_models(self, X, y, test_size=0.2, random_state=42):
        """
        Train multiple regression models on the dataset
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary of trained models
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Train Random Forest
        start_time = time.time()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf_model.fit(X_train, y_train)
        rf_train_time = time.time() - start_time
        
        self.models["Random Forest"] = rf_model
        
        # Train XGBoost
        start_time = time.time()
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
        xgb_model.fit(X_train, y_train)
        xgb_train_time = time.time() - start_time
        
        self.models["XGBoost"] = xgb_model
        
        # Create ensemble (weighted average)
        self.models["Ensemble"] = {
            "Linear Regression": {"model": self.models["Linear Regression"], "weight": 0.2},
            "Random Forest": {"model": rf_model, "weight": 0.3},
            "XGBoost": {"model": xgb_model, "weight": 0.5}
        }
        
        # Calculate and store metrics
        for model_name, model in self.models.items():
            if model_name != "Ensemble":
                y_pred = model.predict(X_test)
                
                self.model_metrics[model_name] = {
                    "r2_score": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "training_time": rf_train_time if model_name == "Random Forest" else xgb_train_time if model_name == "XGBoost" else 0
                }
        
        # Calculate ensemble metrics
        if all(m is not None for m in [self.models["Linear Regression"], self.models["Random Forest"], self.models["XGBoost"]]):
            ensemble_preds = self.predict_with_ensemble(X_test)
            
            self.model_metrics["Ensemble"] = {
                "r2_score": r2_score(y_test, ensemble_preds),
                "mae": mean_absolute_error(y_test, ensemble_preds),
                "rmse": np.sqrt(mean_squared_error(y_test, ensemble_preds)),
                "training_time": rf_train_time + xgb_train_time
            }
        
        # Save models to disk
        self.save_models()
        
        return self.models
    
    def predict_with_model(self, X, model_name="Linear Regression"):
        """
        Make predictions using a specific model
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        model_name : str
            Name of the model to use
            
        Returns:
        --------
        array-like
            Predictions
        """
        if model_name not in self.models or self.models[model_name] is None:
            st.error(f"Model {model_name} not found or not trained yet.")
            return None
        
        if model_name == "Ensemble":
            return self.predict_with_ensemble(X)
        
        return self.models[model_name].predict(X)
    
    def predict_with_ensemble(self, X):
        """
        Make predictions using the ensemble model
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
            
        Returns:
        --------
        array-like
            Weighted average predictions
        """
        if "Ensemble" not in self.models or self.models["Ensemble"] is None:
            st.error("Ensemble model not found or not trained yet.")
            return None
        
        predictions = []
        weights = []
        
        for model_info in self.models["Ensemble"].values():
            model = model_info["model"]
            weight = model_info["weight"]
            
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(weight)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def get_feature_importance(self, model_name="Random Forest"):
        """
        Get feature importance for a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
            
        Returns:
        --------
        array-like
            Feature importance values
        """
        if model_name not in self.models or self.models[model_name] is None:
            st.error(f"Model {model_name} not found or not trained yet.")
            return None
        
        if model_name == "Linear Regression":
            return np.abs(self.models[model_name].coef_)
        elif model_name == "Random Forest":
            return self.models[model_name].feature_importances_
        elif model_name == "XGBoost":
            return self.models[model_name].feature_importances_
        else:
            return None
    
    def get_model_metrics(self, model_name=None):
        """
        Get metrics for a specific model or all models
        
        Parameters:
        -----------
        model_name : str, optional
            Name of the model
            
        Returns:
        --------
        dict
            Model metrics
        """
        if model_name is not None:
            if model_name in self.model_metrics:
                return self.model_metrics[model_name]
            else:
                st.error(f"Metrics for model {model_name} not found.")
                return {}
        
        return self.model_metrics
    
    def preprocess_input(self, input_data):
        """
        Preprocess user input data for prediction
        
        Parameters:
        -----------
        input_data : dict
            User input data
            
        Returns:
        --------
        array-like
            Preprocessed input data
        """
        # Convert categorical features
        if "Extracurricular Activities" in input_data:
            input_data["Extracurricular Activities"] = self.label_encoder.transform([input_data["Extracurricular Activities"]])[0]
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Scale the features
        if self.scaler is not None:
            df_transformed = self.scaler.transform(df)
            return df_transformed
        
        return df.values
    
    def generate_feature_contributions(self, input_data, model_name="Linear Regression"):
        """
        Generate feature contributions to the prediction
        
        Parameters:
        -----------
        input_data : dict
            User input data
        model_name : str
            Name of the model to use
            
        Returns:
        --------
        dict
            Dictionary of feature contributions
        """
        if model_name not in self.models or self.models[model_name] is None:
            st.error(f"Model {model_name} not found or not trained yet.")
            return {}
        
        # For linear regression, we can calculate contributions directly
        if model_name == "Linear Regression":
            model = self.models[model_name]
            preprocessed_data = self.preprocess_input(input_data)
            
            # Calculate baseline (intercept)
            baseline = model.intercept_
            
            # Calculate contributions
            contributions = {}
            for i, feature in enumerate(input_data.keys()):
                contribution = preprocessed_data[0, i] * model.coef_[i]
                contributions[feature] = contribution
            
            return contributions
        
        # For other models, use SHAP values
        elif model_name in ["Random Forest", "XGBoost"]:
            model = self.models[model_name]
            preprocessed_data = self.preprocess_input(input_data)
            
            # Create a background dataset (dummy for this example)
            background_data = np.zeros((1, preprocessed_data.shape[1]))
            
            # SHAP explainer
            if model_name == "Random Forest":
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.TreeExplainer(model)
            
            # Get SHAP values
            shap_values = explainer.shap_values(preprocessed_data)
            
            # Calculate contributions
            contributions = {}
            for i, feature in enumerate(input_data.keys()):
                contributions[feature] = shap_values[0, i]
            
            return contributions
        
        return {} 