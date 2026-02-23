"""
ADVANCED GUI for AQI Prediction Project
Run with: streamlit run gui_advanced.py
"""
import streamlit as st
# ADD THIS LINE RIGHT AFTER:
st.set_page_config(page_title="AQI Prediction", page_icon="📊", layout="wide")
# Then remove the existing st.set_page_config (line 14-18) - it will be duplicate

import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import sys      # <---- ADD THIS LINE
import os 

sys.path.append(os.getcwd())

# Page config
st.set_page_config(
    page_title="AQI Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .health-good { color: #00C853; }
    .health-moderate { color: #FFD600; }
    .health-unhealthy { color: #FF9100; }
    .health-very-unhealthy { color: #FF3D00; }
    .health-hazardous { color: #B71C1C; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 AQI Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
    *Real-time Air Quality Index Prediction System using Machine Learning*
    """)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    city_options = {
        "Delhi": {"lat": 28.6139, "lon": 77.2090},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777},
        "Chennai": {"lat": 13.0827, "lon": 80.2707},
        "Kolkata": {"lat": 22.5726, "lon": 88.3639},
        "Bangalore": {"lat": 12.9716, "lon": 77.5946}
    }
    
    selected_city = st.selectbox("🌍 Select City", list(city_options.keys()))
    
    # Auto-refresh option
    auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=False)
    
    st.markdown("---")
    
    # Model info
    st.subheader("🤖 Model Information")
    st.metric("Accuracy (R²)", "99.71%")
    st.metric("Average Error", "±1.67 AQI")
    st.metric("Training Samples", "1,440")
    
    st.markdown("---")
    
    # Project info
    st.subheader("🎓 Project Info")
    st.info("""
        **College Project:** AQI Prediction using ML  
        **Tech Stack:** Python, XGBoost, Streamlit  
        **Data Source:** OpenWeatherMap API  
        **Features:** Real-time prediction, Health advisory
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📈 Prediction", "📊 Analysis", "ℹ️ About"])

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.button("📡 Fetch Live Data", key="fetch_btn", use_container_width=True):
            try:
                from config import OPENWEATHER_API_KEY
                city_data = city_options[selected_city]
                
                url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={city_data['lat']}&lon={city_data['lon']}&appid={OPENWEATHER_API_KEY}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state['api_data'] = data
                    st.session_state['last_fetch'] = datetime.now()
                    st.success("✅ Data fetched successfully!")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.button("🤖 Predict AQI", key="predict_btn", use_container_width=True, disabled='api_data' not in st.session_state):
            try:
                from predict_aqi_simple import predict_aqi
                
                data = st.session_state['api_data']
                pollutants = data['list'][0]['components']
                now = datetime.now()
                
                prediction_data = {
                    'PM2.5': pollutants['pm2_5'],
                    'PM10': pollutants['pm10'],
                    'O3': pollutants['o3'],
                    'NO2': pollutants['no2'],
                    'SO2': pollutants['so2'],
                    'CO': pollutants['co'],
                    'hour': now.hour,
                    'day_of_week': now.weekday(),
                    'month': now.month,
                    'is_rush_hour': 1 if (7 <= now.hour <= 10) or (17 <= now.hour <= 20) else 0
                }
                
                predicted_aqi = predict_aqi(prediction_data)
                st.session_state['predicted_aqi'] = predicted_aqi
                st.session_state['prediction_time'] = now
                st.success(f"✅ Predicted AQI: {predicted_aqi:.1f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'last_fetch' in st.session_state:
            st.metric("Last Fetch", st.session_state['last_fetch'].strftime("%H:%M:%S"))
        else:
            st.metric("Last Fetch", "Never")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    if 'api_data' in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Current Pollution Levels")
            
            data = st.session_state['api_data']
            pollutants = data['list'][0]['components']
            
            # Create bar chart
            poll_names = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
            poll_values = [
                pollutants['pm2_5'], pollutants['pm10'], pollutants['o3'],
                pollutants['no2'], pollutants['so2'], pollutants['co']
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=poll_names,
                    y=poll_values,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
                )
            ])
            
            fig.update_layout(
                title="Pollutant Concentrations (μg/m³)",
                yaxis_title="Concentration",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            df_poll = pd.DataFrame({
                'Pollutant': poll_names,
                'Value': poll_values,
                'Unit': 'μg/m³'
            })
            st.dataframe(df_poll, use_container_width=True)
        
        with col2:
            st.subheader("🔮 AQI Prediction")
            
            if 'predicted_aqi' in st.session_state:
                aqi = st.session_state['predicted_aqi']
                
                # AQI Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=aqi,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted AQI"},
                    gauge={
                        'axis': {'range': [0, 500]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "green"},
                            {'range': [51, 100], 'color': "yellow"},
                            {'range': [101, 150], 'color': "orange"},
                            {'range': [151, 200], 'color': "red"},
                            {'range': [201, 300], 'color': "purple"},
                            {'range': [301, 500], 'color': "maroon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': aqi
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Health advisory
                st.subheader("💊 Health Advisory")
                
                if aqi <= 50:
                    st.success("""
                        **✅ GOOD** - Air quality is satisfactory.
                        * No health implications.
                        * Ideal for outdoor activities.
                    """)
                elif aqi <= 100:
                    st.info("""
                        **⚠️ MODERATE** - Acceptable air quality.
                        * Unusually sensitive people should consider limiting outdoor activities.
                        * General public is not likely to be affected.
                    """)
                elif aqi <= 150:
                    st.warning("""
                        **😷 UNHEALTHY FOR SENSITIVE GROUPS**
                        * People with lung disease, children, older adults should reduce outdoor activities.
                        * General public usually not affected.
                    """)
                elif aqi <= 200:
                    st.warning("""
                        **🚨 UNHEALTHY**
                        * Everyone may begin to experience health effects.
                        * Sensitive groups should avoid outdoor activities.
                    """)
                elif aqi <= 300:
                    st.error("""
                        **🆘 VERY UNHEALTHY**
                        * Health alert: everyone may experience more serious health effects.
                        * Avoid all outdoor activities.
                    """)
                else:
                    st.error("""
                        **☠️ HAZARDOUS**
                        * Emergency conditions.
                        * Entire population is likely to be affected.
                        * Stay indoors with air purifiers.
                    """)
                
                # Prediction time
                if 'prediction_time' in st.session_state:
                    st.caption(f"Predicted at: {st.session_state['prediction_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.info("Click 'Predict AQI' button to see results")
    else:
        st.info("👈 Fetch live data first from the Dashboard tab!")

with tab3:
    st.subheader("📈 Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance chart
        st.subheader("🔍 Feature Importance")
        
        importance_data = {
            'Feature': ['PM10', 'PM2.5', 'O3', 'SO2', 'is_rush_hour', 'NO2', 'CO', 'day_of_week', 'hour', 'month'],
            'Importance': [0.7361, 0.2189, 0.0427, 0.0006, 0.0005, 0.0004, 0.0003, 0.0003, 0.0002, 0.0001]
        }
        
        df_importance = pd.DataFrame(importance_data)
        
        fig_importance = px.bar(
            df_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='ML Model Feature Importance',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.subheader("📊 Model Performance")
        
        metrics_data = {
            'Metric': ['R² Score', 'Mean Absolute Error', 'Root Mean Squared Error'],
            'Value': ['99.71%', '1.67 AQI points', '2.49'],
            'Description': ['Accuracy', 'Average Error', 'Standard Deviation of Error']
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Performance notes
        st.info("""
            **Performance Notes:**
            - R² > 0.99 indicates excellent predictive power
            - MAE < 2 means average error is less than 2 AQI points
            - Model trained on 1,440 samples with time-series validation
        """)

with tab4:
    st.subheader("🎓 About This Project")
    
    st.markdown("""
    ### **AQI Prediction System using Machine Learning**
    
    **📋 Project Overview:**
    This system predicts the Air Quality Index (AQI) in real-time using machine learning.
    It fetches live pollution data from APIs and uses a trained XGBoost model to predict AQI.
    
    **🎯 Objectives:**
    1. Real-time air quality monitoring
    2. Accurate AQI prediction using ML
    3. Health advisory generation
    4. User-friendly visualization
    
    **🛠️ Technical Implementation:**
    - **Data Source:** OpenWeatherMap API (real-time pollution data)
    - **ML Model:** XGBoost Regressor (99.71% accuracy)
    - **Frontend:** Streamlit web interface
    - **Backend:** Python, Pandas, Scikit-learn
    
    **📊 Key Features:**
    - Real-time data fetching
    - ML-powered AQI prediction
    - Interactive visualizations
    - Health recommendations
    - Multi-city support
    
    **👨‍💻 Developed By:** [Your Name]
    **🎓 College Project** - Computer Science/Engineering
    
    **📁 Project Structure:**
    ```
    aqi_project/
    ├── ml_final.py           # ML model training
    ├── predict_aqi_simple.py # Prediction function
    ├── test_api.py          # API data fetching
    ├── gui_advanced.py      # This GUI
    ├── models/              # Trained ML models
    └── data/               # Datasets
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>AQI Prediction using Machine Learning | Accuracy: 99.71%</p>
        <p style='color: gray; font-size: 0.9rem'>Refresh page for latest data | Data source: OpenWeatherMap API</p>
    </div>
""", unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh and 'last_fetch' in st.session_state:
    if datetime.now() - st.session_state['last_fetch'] > timedelta(seconds=30):
        st.rerun()