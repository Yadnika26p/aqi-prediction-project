"""
AQI Prediction Function with Fallback
"""
import joblib
import pandas as pd
import numpy as np
import streamlit as st

def predict_aqi(current_data):
    """
    Predict AQI based on current pollution levels
    """
    try:
        # Try to load XGBoost model
        try:
            model = joblib.load('models/aqi_predictor_model.pkl')
            scaler = joblib.load('models/aqi_scaler.pkl')
            
            with open('models/feature_names.txt', 'r') as f:
                expected_features = [line.strip() for line in f.readlines()]
            
            features_df = pd.DataFrame([current_data])
            
            for feature in expected_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            features_df = features_df[expected_features]
            scaled_features = scaler.transform(features_df)
            predicted_aqi = model.predict(scaled_features)[0]
            
            return round(float(predicted_aqi), 2)
            
        except Exception as e:
            st.warning(f"ML Model error: {e}. Using fallback calculation.")
            # Fallback: Simple weighted AQI calculation
            if 'PM2.5' in current_data and 'PM10' in current_data:
                # Simple formula based on EPA conversion
                pm25 = current_data['PM2.5']
                pm10 = current_data['PM10']
                
                # Rough AQI calculation (simplified)
                aqi_pm25 = pm25 * 1.0  # Simplified conversion
                aqi_pm10 = pm10 * 0.5   # Simplified conversion
                
                predicted_aqi = max(aqi_pm25, aqi_pm10)
                return round(float(predicted_aqi), 2)
            return 100.0  # Default
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return 150.0  # Default moderate AQI

# For testing
if __name__ == "__main__":
    sample_data = {
        'PM2.5': 106.55,
        'PM10': 150.64,
        'O3': 137.60,
        'NO2': 3.41,
        'SO2': 4.98,
        'CO': 472.51,
        'hour': 14,
        'day_of_week': 0,
        'month': 2,
        'is_rush_hour': 0
    }
    
    aqi = predict_aqi(sample_data)
    print(f"Predicted AQI: {aqi}")