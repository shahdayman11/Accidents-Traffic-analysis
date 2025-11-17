import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from catboost import CatBoostClassifier
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dual Accident Prediction System",
    page_icon="🚗", 
    layout="wide"
)

# Only download LARGE files, include small ones in repo
SMALL_MODELS = ['scaler.pkl', 'power_transformer.pkl', 'feature_names.pkl', 
                'preprocessing_objects.pkl', 'risk_model_info.pkl']

LARGE_MODELS = {
    'severity_gb_model.pkl': 'https://drive.google.com/uc?export=download&id=1e_QQLqisbiaucI1PvGZS_NbJ6tfqVnVL',
    'feature_selector.pkl': 'https://drive.google.com/uc?export=download&id=1UAs3iGBtKVQeMQ6FUh6cb0_bzF_4ul5J',
    'risk_catboost_model.cbm': 'https://drive.google.com/uc?export=download&id=1TrgEU86-KZ5-V8m8AbNLcyCUM9exAllb'
}

def download_large_file(url, filename):
    """Download large files with progress"""
    try:
        # Use gdown library for Google Drive
        import gdown
        output = f'models/{filename}'
        gdown.download(url, output, quiet=False)
        return os.path.exists(output)
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False

@st.cache_resource
def load_models():
    """Load models - small ones from repo, large ones from download"""
    os.makedirs('models', exist_ok=True)
    
    # Copy small models from repo to models folder
    for model_file in SMALL_MODELS:
        if os.path.exists(model_file):
            import shutil
            shutil.copy(model_file, f'models/{model_file}')
        else:
            st.warning(f"Small model file {model_file} not found in repo")
    
    # Download large models
    for filename, url in LARGE_MODELS.items():
        filepath = f'models/{filename}'
        if not os.path.exists(filepath):
            with st.spinner(f"📥 Downloading large file {filename}..."):
                if not download_large_file(url, filename):
                    return None
    
    try:
        # Load all models
        severity_model = joblib.load('models/severity_gb_model.pkl')
        feature_selector = joblib.load('models/feature_selector.pkl')
        scaler = joblib.load('models/scaler.pkl')
        power_transformer = joblib.load('models/power_transformer.pkl')
        
        with open('models/feature_names.pkl', 'rb') as f:
            severity_features = pickle.load(f)
            
        preprocessing_objects = joblib.load('models/preprocessing_objects.pkl')
        
        risk_model = CatBoostClassifier()
        risk_model.load_model('models/risk_catboost_model.cbm')
        risk_model_info = joblib.load('models/risk_model_info.pkl')
        
        st.success("✅ All models loaded successfully!")
        return (severity_model, feature_selector, scaler, power_transformer,
                severity_features, preprocessing_objects, risk_model, risk_model_info)
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# Initialize the app
st.title("🚗 Dual Accident Prediction System")

# Load models
with st.spinner("Loading machine learning models..."):
    models = load_models()

if models is None:
    st.error("""
    **Failed to load models. Please:**
    1. Make sure the small model files are in your GitHub repo
    2. Check that Google Drive links are working
    3. Try refreshing the app
    """)
    st.stop()

# Unpack models
(severity_model, feature_selector, scaler, power_transformer,
 severity_features, preprocessing_objects, risk_model, risk_model_info) = models

severity_threshold = preprocessing_objects['severity_threshold']
skewed_features = preprocessing_objects['skewed_features']
risk_num_cols = risk_model_info['num_cols']
risk_cat_cols = risk_model_info['cat_cols']
risk_threshold = risk_model_info['risk_threshold']

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Prediction Mode", 
                               ["Severity Prediction", "Risk Prediction", "About"])

if app_mode == "Severity Prediction":
    st.header("📊 Accident Severity Prediction")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Location & Time")
        month = st.slider("Month", 1, 12, 6)
        start_lat = st.number_input("Start Latitude", value=37.77, format="%.6f")
        start_lng = st.number_input("Start Longitude", value=-122.42, format="%.6f")
        distance = st.number_input("Distance (miles)", value=1.5, min_value=0.0)
        delay_typical = st.number_input("Delay from Typical Traffic (mins)", value=15.0, min_value=0.0)

    with col2:
        st.subheader("Weather Conditions")
        temperature = st.number_input("Temperature (°F)", value=65.0)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        pressure = st.number_input("Pressure (inHg)", value=29.92)
        visibility = st.number_input("Visibility (miles)", value=10.0, min_value=0.0)
        wind_speed = st.number_input("Wind Speed (mph)", value=8.0, min_value=0.0)

    with col3:
        st.subheader("Traffic & Accident Data")
        accident_count = st.number_input("Accident Count", value=5, min_value=0)
        traffic_event_count = st.number_input("Traffic Event Count", value=3, min_value=0)
        traffic_severity = st.slider("Traffic Severity", 1, 5, 3)
        delay_freeflow = st.number_input("Delay from Free Flow (mins)", value=8.0, min_value=0.0)
        precipitation = st.number_input("Precipitation (inches)", value=0.0, min_value=0.0)

    # Additional parameters
    st.subheader("Additional Parameters")
    col4, col5 = st.columns(2)
    with col4:
        is_rush_hour = st.selectbox("Is Rush Hour?", ["No", "Yes"])
        is_weekend = st.selectbox("Is Weekend?", ["No", "Yes"])
    with col5:
        season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])

    if st.button("🔍 Predict Severity", type="primary", use_container_width=True):
        # Convert inputs
        rush_hour_val = 1 if is_rush_hour == "Yes" else 0
        weekend_val = 1 if is_weekend == "Yes" else 0
        season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
        season_val = season_mapping[season]

        # Create input data
        input_data = pd.DataFrame({
            'Month': [month],
            'Accident_Count': [accident_count],
            'Start_Lat': [start_lat],
            'Start_Lng': [start_lng],
            'Distance(mi)': [distance],
            'Temperature(F)': [temperature],
            'Humidity(%)': [humidity],
            'Pressure(in)': [pressure],
            'Visibility(mi)': [visibility],
            'Wind_Speed(mph)': [wind_speed],
            'Traffic_Event_Count': [traffic_event_count],
            'Traffic_Severity': [traffic_severity],
            'DelayFromTypicalTraffic(mins)': [delay_typical],
            'DelayFromFreeFlowSpeed(mins)': [delay_freeflow],
            'Precipitation(in)': [precipitation],
            'Is_Rush_Hour': [rush_hour_val],
            'Weather_Severity_Index': [
                (10 - min(visibility, 10)) * 0.3 + 
                np.log1p(wind_speed) * 0.3 + 
                np.log1p(precipitation * 50) * 0.2 + 
                (100 - humidity) * 0.2
            ],
            'Traffic_Congestion_Score': [
                np.log1p(delay_typical) * 0.5 + 
                np.log1p(delay_freeflow) * 0.3 + 
                np.log1p(traffic_event_count) * 0.2
            ],
            'Accident_Density': [max(accident_count, 1)],
            'Log_Accident_Density': [np.log1p(max(accident_count, 1))],
            'Weather_Traffic_Interaction': [0],
            'Visibility_Humidity_Interaction': [visibility * humidity],
            'Temperature_Squared': [temperature ** 2],
            'Wind_Speed_Squared': [wind_speed ** 2],
            'Season': [season_val],
            'Is_Weekend': [weekend_val]
        })

        input_data['Weather_Traffic_Interaction'] = (
            input_data['Weather_Severity_Index'] * input_data['Traffic_Congestion_Score']
        )

        try:
            # Preprocess and predict
            input_skewed = power_transformer.transform(input_data[skewed_features])
            input_transformed = input_data.copy()
            for i, col in enumerate(skewed_features):
                input_transformed[col] = input_skewed[:, i]

            input_selected = feature_selector.transform(input_transformed)
            prediction = severity_model.predict(input_selected)[0]
            probability = severity_model.predict_proba(input_selected)[0]

            # Display results
            st.markdown("---")
            col_result1, col_result2 = st.columns([1, 2])
            
            with col_result1:
                if prediction == 1:
                    st.error(f"🚨 HIGH SEVERITY ACCIDENT")
                    st.metric("Probability", f"{probability[1]:.1%}")
                else:
                    st.success(f"✅ LOW SEVERITY ACCIDENT") 
                    st.metric("Probability", f"{probability[0]:.1%}")
            
            with col_result2:
                fig, ax = plt.subplots(figsize=(8, 2))
                colors = ['#28a745', '#dc3545']
                labels = ['Low Severity', 'High Severity']
                ax.barh(labels, probability, color=colors)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Confidence')
                for i, v in enumerate(probability):
                    ax.text(v + 0.01, i, f'{v:.1%}', va='center')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

elif app_mode == "Risk Prediction":
    st.header("🔥 Risk Prediction")
    st.info("Risk prediction feature coming soon...")
    st.write("This will predict high-risk accident areas using the CatBoost model.")

else:
    st.header("📖 About This System")
    st.markdown("""
    ## Dual Accident Prediction System
    
    **Features:**
    - 🚨 **Severity Prediction**: Gradient Boosting model for accident severity
    - 🔥 **Risk Prediction**: CatBoost model for high-risk areas
    - ☁️ **Hybrid Deployment**: Small models in repo, large models from Google Drive
    
    **Model Status:** ✅ Loaded Successfully
    """)

st.markdown("---")
st.markdown("Deployed on Streamlit Community Cloud | Hybrid Model Loading")
