import streamlit as st
import requests
import json
from streamlit_lottie import st_lottie
from PIL import Image
import base64
import plotly.graph_objects as go
import numpy as np

def set_page_config():
    """Configure the Streamlit page layout and style"""
    st.set_page_config(
        page_title="Student Performance Analytics",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2d4db1;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #1e3a8a;
    }
    </style>
    """, unsafe_allow_html=True)

def load_lottie_url(url):
    """Load animation from Lottie URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            st.warning(f"Failed to load animation from URL: {url}")
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Error loading Lottie animation: {e}")
        return None

def load_lottie_fallback():
    """Provide a fallback Lottie animation when URL loading fails"""
    # Simple loading spinner animation in JSON format
    return {
        "v": "5.7.1",
        "fr": 30,
        "ip": 0,
        "op": 60,
        "w": 200,
        "h": 200,
        "nm": "Loading Spinner",
        "ddd": 0,
        "assets": [],
        "layers": [{
            "ddd": 0,
            "ind": 1,
            "ty": 4,
            "nm": "Spinner",
            "sr": 1,
            "ks": {
                "o": {"a": 0, "k": 100},
                "r": {"a": 1, "k": [{"t": 0, "s": [0]}, {"t": 60, "s": [360]}]},
                "p": {"a": 0, "k": [100, 100]},
                "a": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [100, 100]}
            },
            "shapes": [{
                "ty": "el",
                "p": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [80, 80]},
                "d": 1,
                "nm": "Ellipse Path 1",
                "mn": "ADBE Vector Shape - Ellipse"
            }, {
                "ty": "st",
                "c": {"a": 0, "k": [0.12, 0.23, 0.54, 1]},
                "o": {"a": 0, "k": 100},
                "w": {"a": 0, "k": 10},
                "d": [{"n": "d", "v": {"a": 0, "k": 20}}, {"n": "g", "v": {"a": 0, "k": 20}}],
                "lc": 2,
                "lj": 1,
                "ml": 4,
                "nm": "Stroke 1",
                "mn": "ADBE Vector Graphic - Stroke"
            }, {
                "ty": "tr",
                "p": {"a": 0, "k": [0, 0]},
                "a": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [100, 100]},
                "r": {"a": 0, "k": 0},
                "o": {"a": 0, "k": 100}
            }]
        }]
    }

def display_lottie_animation(lottie_url=None, height=300, width=700):
    """Display a Lottie animation in the Streamlit app"""
    lottie_json = None
    
    # Try to load from URL if provided
    if lottie_url:
        lottie_json = load_lottie_url(lottie_url)
        
    # Use fallback if URL loading failed
    if not lottie_json:
        lottie_json = load_lottie_fallback()
        
    try:
        st_lottie(lottie_json, height=height, width=width, key="lottie")
    except Exception as e:
        st.warning(f"Failed to display animation: {e}")
        # Display a simple placeholder
        st.markdown(f"""
        <div style="height:{height}px; width:{width}px; 
                   background-color:#f0f2f6; border-radius:10px; 
                   display:flex; align-items:center; justify-content:center;">
            <p style="color:#1e3a8a; font-size:20px;">📊 Analytics Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

def get_img_as_base64(file):
    """Convert image to base64 string for CSS background"""
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def create_header():
    """Create an attractive header for the application"""
    st.markdown("""
    <div style="display: flex; align-items: center; background-color: #1e3a8a; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <div style="margin-right: 2rem;">
            <h1 style="color: white; margin: 0;">Student Performance Analytics</h1>
            <p style="color: #d1d5db; margin: 0; font-size: 1.1rem;">Advanced prediction platform with ML ensemble techniques</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_header():
    """Create a styled sidebar header"""
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #1e3a8a;">⚙️ Control Panel</h2>
        <p style="color: #4b5563;">Configure model parameters and view analytics</p>
    </div>
    """, unsafe_allow_html=True)

def create_info_box(title, content, icon="ℹ️"):
    """Create an information box with styled content"""
    st.markdown(f"""
    <div style="background-color: #e0f2fe; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
        <h3 style="margin-top: 0; color: #1e3a8a; display: flex; align-items: center;">
            <span style="margin-right: 0.5rem;">{icon}</span> {title}
        </h3>
        <p style="margin-bottom: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def display_metrics_dashboard(model_metrics, model_name="Model"):
    """Display a metrics dashboard for model performance"""
    st.markdown(f"### {model_name} Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="R² Score",
            value=f"{model_metrics.get('r2_score', 0):.3f}",
            delta=f"{model_metrics.get('r2_score_change', 0):.2f}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="MAE",
            value=f"{model_metrics.get('mae', 0):.3f}",
            delta=f"{model_metrics.get('mae_change', 0):.2f}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="RMSE",
            value=f"{model_metrics.get('rmse', 0):.3f}",
            delta=f"{model_metrics.get('rmse_change', 0):.2f}",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Training Time",
            value=f"{model_metrics.get('training_time', 0):.2f}s",
            delta=None
        )

def create_gauge_chart(value, title, min_val=0, max_val=100, threshold_ranges=None):
    """
    Create a gauge chart for displaying prediction results
    
    Parameters:
    -----------
    value : float
        Value to display on the gauge
    title : str
        Title of the gauge chart
    min_val : int
        Minimum value on the gauge
    max_val : int
        Maximum value on the gauge
    threshold_ranges : list of dicts, optional
        List of dictionaries with 'min', 'max', 'color', and 'label' keys
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Gauge chart figure
    """
    if threshold_ranges is None:
        threshold_ranges = [
            {"min": 0, "max": 40, "color": "red", "label": "Poor"},
            {"min": 40, "max": 60, "color": "orange", "label": "Average"},
            {"min": 60, "max": 75, "color": "yellow", "label": "Good"},
            {"min": 75, "max": 90, "color": "lightgreen", "label": "Very Good"},
            {"min": 90, "max": 100, "color": "green", "label": "Excellent"}
        ]
    
    steps = []
    for range_info in threshold_ranges:
        steps.append({
            "range": [range_info["min"], range_info["max"]],
            "color": range_info["color"]
        })
    
    # Determine the threshold color based on value
    current_threshold = None
    for range_info in threshold_ranges:
        if range_info["min"] <= value <= range_info["max"]:
            current_threshold = range_info
            break
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 24}},
        delta={"reference": 75, "increasing": {"color": "green"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": current_threshold["color"] if current_threshold else "blue"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": steps,
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={"color": "darkblue", "family": "Arial"}
    )
    
    return fig

def create_comparison_radar_chart(student_values, average_values, feature_names):
    """
    Create a radar chart comparing student values to average values
    
    Parameters:
    -----------
    student_values : list
        Student's values for each feature
    average_values : list
        Average values for each feature
    feature_names : list
        Names of the features
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Radar chart figure
    """
    fig = go.Figure()
    
    # Repeat the first value to close the loop
    student_values_closed = student_values + [student_values[0]]
    average_values_closed = average_values + [average_values[0]]
    feature_names_closed = feature_names + [feature_names[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=student_values_closed,
        theta=feature_names_closed,
        fill='toself',
        name='Your Inputs',
        line=dict(color='rgba(31, 119, 180, 0.8)', width=2),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=average_values_closed,
        theta=feature_names_closed,
        fill='toself',
        name='Average Student',
        line=dict(color='rgba(255, 127, 14, 0.8)', width=2),
        fillcolor='rgba(255, 127, 14, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(student_values), max(average_values)) * 1.2]
            )
        ),
        showlegend=True,
        title="Your Profile vs. Average Student",
        height=500
    )
    
    return fig

def create_prediction_progress_bars(prediction_breakdown):
    """
    Display prediction breakdown as progress bars
    
    Parameters:
    -----------
    prediction_breakdown : dict
        Dictionary with feature names as keys and contribution values as values
    """
    st.markdown("### Prediction Breakdown")
    st.markdown("This shows how each factor contributes to your predicted score:")
    
    total = sum(abs(val) for val in prediction_breakdown.values())
    
    for feature, contribution in sorted(prediction_breakdown.items(), key=lambda x: abs(x[1]), reverse=True):
        percentage = (abs(contribution) / total) * 100
        color = "green" if contribution > 0 else "red"
        
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span>{feature}</span>
                <span style="color: {'green' if contribution > 0 else 'red'};">
                    {"+" if contribution > 0 else ""}{contribution:.2f} points
                </span>
            </div>
            <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px;">
                <div style="width: {percentage}%; background-color: {color}; height: 10px; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def generate_recommendations(user_data, prediction, threshold=75):
    """
    Generate personalized recommendations based on user data and prediction
    
    Parameters:
    -----------
    user_data : dict
        User input data
    prediction : float
        Predicted performance score
    threshold : float
        Desired performance threshold
        
    Returns:
    --------
    list
        List of recommendation strings
    """
    recommendations = []
    
    # Hours studied recommendation
    if user_data["Hours Studied"] < 6:
        needed_hours = min(10, user_data["Hours Studied"] + 2)
        recommendations.append(f"Increase study time from {user_data['Hours Studied']} to {needed_hours} hours daily to potentially improve performance.")
    
    # Sleep hours recommendation
    if user_data["Sleep Hours"] < 7:
        recommendations.append(f"Consider getting more sleep (aim for 7-8 hours). Current: {user_data['Sleep Hours']} hours.")
    elif user_data["Sleep Hours"] > 9:
        recommendations.append(f"You might benefit from slightly less sleep (7-8 hours optimal). Current: {user_data['Sleep Hours']} hours.")
    
    # Practice papers recommendation
    if user_data["Sample Question Papers Practiced"] < 5:
        recommendations.append(f"Practice more sample question papers. Increase from {user_data['Sample Question Papers Practiced']} to at least 5.")
    
    # Extracurricular recommendation
    if user_data["Extracurricular Activities"] == "No":
        recommendations.append("Consider participating in extracurricular activities that don't significantly reduce study time. They can help with overall cognitive development.")
    
    # If predicted score is already high
    if prediction >= threshold:
        recommendations.append(f"Your predicted score ({prediction:.1f}) is already above the target threshold ({threshold}). Keep up your current study habits!")
    
    return recommendations 