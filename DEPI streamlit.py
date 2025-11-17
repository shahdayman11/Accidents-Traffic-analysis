import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
import joblib
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Accident Risk Prediction System",
    page_icon="🚨", 
    layout="wide"
)

# Google Drive URLs for Risk Model
RISK_MODEL_URLS = {
    'risk_catboost_model.cbm': 'https://drive.google.com/uc?export=download&id=1TrgEU86-KZ5-V8m8AbNLcyCUM9exAllb',
    'risk_model_info.pkl': 'https://drive.google.com/uc?export=download&id=1uMtB3ik4j1gNoZ8G9XCKIG-NwqIGk5G5',
    'risk_location_plot.png': 'https://drive.google.com/uc?export=download&id=YOUR_LOCATION_PLOT_ID',
    'feature_importance_plot.png': 'https://drive.google.com/uc?export=download&id=YOUR_IMPORTANCE_PLOT_ID'
}

def download_file_simple(url, filename):
    """Simple download function"""
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
        else:
            st.error(f"HTTP Error {response.status_code} for {filename}")
            return False
    except Exception as e:
        st.error(f"Error downloading {filename}: {str(e)}")
        return False

@st.cache_resource
def load_risk_model():
    """Load risk prediction model"""
    os.makedirs('models', exist_ok=True)
    
    # Download model files
    for filename, url in RISK_MODEL_URLS.items():
        filepath = f'models/{filename}'
        if not os.path.exists(filepath):
            with st.spinner(f"📥 Downloading {filename}..."):
                if not download_file_simple(url, filepath):
                    return None
    
    try:
        # Load risk model
        risk_model = CatBoostClassifier()
        risk_model.load_model('models/risk_catboost_model.cbm')
        risk_model_info = joblib.load('models/risk_model_info.pkl')
        
        st.success("✅ Risk model loaded successfully!")
        return risk_model, risk_model_info
        
    except Exception as e:
        st.error(f"❌ Error loading risk model: {str(e)}")
        return None

# Initialize the app
st.title("🚨 Accident Risk Prediction System")

# Load model
model_data = load_risk_model()

if model_data is None:
    st.error("Failed to load risk model. Please check your Google Drive links.")
    st.stop()

risk_model, risk_model_info = model_data

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Section", 
                               ["Risk Prediction", "Model Insights", "About"])

if app_mode == "Risk Prediction":
    st.header("🔮 Predict High-Risk Areas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location Information")
        state = st.text_input("State", "CA")
        county = st.text_input("County", "Los Angeles")
        city = st.text_input("City", "Los Angeles")
        latitude = st.number_input("Latitude", value=34.05, format="%.6f")
        longitude = st.number_input("Longitude", value=-118.24, format="%.6f")
        year = st.number_input("Year", value=2024, min_value=2000, max_value=2030)
        month = st.slider("Month", 1, 12, 6)
        
    with col2:
        st.subheader("Environmental & Traffic Data")
        temperature = st.number_input("Temperature (°F)", value=75.0)
        humidity = st.slider("Humidity (%)", 0, 100, 45)
        pressure = st.number_input("Pressure (in)", value=29.92)
        visibility = st.number_input("Visibility (miles)", value=10.0)
        wind_speed = st.number_input("Wind Speed (mph)", value=8.0)
        precipitation = st.number_input("Precipitation (inches)", value=0.0)
        traffic_event_count = st.number_input("Traffic Event Count", value=5)
        traffic_severity = st.slider("Traffic Severity", 1, 5, 3)
        delay_typical = st.number_input("Delay from Typical Traffic (mins)", value=12.0)
        delay_freeflow = st.number_input("Delay from Free Flow (mins)", value=6.0)
        traffic_distance = st.number_input("Traffic Distance", value=2.5)

    if st.button("🎯 Predict Risk Level", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'Year': [year],
            'Month': [month],
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Temperature(F)': [temperature],
            'Humidity(%)': [humidity],
            'Pressure(in)': [pressure],
            'Visibility(mi)': [visibility],
            'Wind_Speed(mph)': [wind_speed],
            'Precipitation(in)': [precipitation],
            'Traffic_Event_Count': [traffic_event_count],
            'Traffic_Severity': [traffic_severity],
            'DelayFromTypicalTraffic(mins)': [delay_typical],
            'DelayFromFreeFlowSpeed(mins)': [delay_freeflow],
            'Traffic_Distance': [traffic_distance],
            'State': [state],
            'County': [county],
            'City': [city]
        })
        
        try:
            # Get categorical feature indices
            cat_features = [input_data.columns.get_loc(col) for col in risk_model_info['cat_cols']]
            
            # Predict
            prediction = risk_model.predict(input_data)[0]
            probability = risk_model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            col_result1, col_result2 = st.columns([1, 2])
            
            with col_result1:
                if prediction == 1:
                    st.error(f"🔥 HIGH RISK AREA")
                    st.metric("Risk Probability", f"{probability[1]:.1%}")
                    st.info("""
                    **High Risk Indicators:**
                    - Potential for severe accidents
                    - Consider safety measures
                    - Monitor conditions closely
                    """)
                else:
                    st.success(f"✅ LOW RISK AREA")
                    st.metric("Risk Probability", f"{probability[0]:.1%}")
                    st.info("""
                    **Low Risk Area:**
                    - Normal safety protocols sufficient
                    - Continue monitoring
                    """)
            
            with col_result2:
                fig, ax = plt.subplots(figsize=(8, 2))
                colors = ['#28a745', '#dc3545']
                labels = ['Low Risk', 'High Risk']
                ax.barh(labels, probability, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Risk Prediction Confidence')
                for i, v in enumerate(probability):
                    ax.text(v + 0.01, i, f'{v:.1%}', va='center')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

elif app_mode == "Model Insights":
    st.header("📊 Model Insights & Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        try:
            st.image('models/risk_distribution_plot.png', 
                    caption='Distribution of High vs Low Risk Areas')
        except:
            st.info("Risk distribution plot not available")
        
        st.subheader("Feature Importance")
        try:
            st.image('models/feature_importance_plot.png',
                    caption='Top 15 Most Important Features for Risk Prediction')
        except:
            st.info("Feature importance plot not available")
    
    with col2:
        st.subheader("Geographical Risk Patterns")
        try:
            st.image('models/risk_location_plot.png',
                    caption='Accident Risk by Geographical Location')
        except:
            st.info("Location plot not available")
        
        st.subheader("Risk Score Distribution")
        try:
            st.image('models/risk_score_distribution.png',
                    caption='Distribution of Calculated Risk Scores')
        except:
            st.info("Risk score distribution plot not available")

else:
    st.header("📖 About This System")
    st.markdown("""
    ## Accident Risk Prediction System
    
    **Purpose:**
    - Predict high-risk accident areas using machine learning
    - Help prioritize safety measures and resources
    - Provide actionable insights for accident prevention
    
    **Model Details:**
    - **Algorithm**: CatBoost Classifier
    - **Features**: 18 total (15 numerical, 3 categorical)
    - **Target**: High_Risk (binary classification)
    
    **Key Features Used:**
    - Location data (State, County, City, Coordinates)
    - Weather conditions
    - Traffic patterns and severity
    - Historical accident data
    
    **Model Status:** ✅ Loaded Successfully
    """)

st.markdown("---")
st.markdown("Built with CatBoost | Deployed on Streamlit Cloud")
