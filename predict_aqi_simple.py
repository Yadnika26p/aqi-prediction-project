"""
AQI Prediction Function
Usage: predict_aqi(current_data) returns predicted AQI
"""
import joblib
import pandas as pd
import numpy as np

def predict_aqi(current_data):
    """
    Predict AQI based on current pollution levels
    
    Args:
        current_data: Dictionary with current readings
            Example: {
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
    
    Returns:
        Predicted AQI (float)
    """
    try:
        # Load model and scaler
        model = joblib.load('models/aqi_predictor_model.pkl')
        scaler = joblib.load('models/aqi_scaler.pkl')
        
        # Load expected feature names
        with open('models/feature_names.txt', 'r') as f:
            expected_features = [line.strip() for line in f.readlines()]
        
        # Create DataFrame with correct feature order
        features_df = pd.DataFrame([current_data])
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0  # Fill missing with 0
        
        # Reorder columns to match training
        features_df = features_df[expected_features]
        
        # Scale features
        scaled_features = scaler.transform(features_df)
        
        # Predict
        predicted_aqi = model.predict(scaled_features)[0]
        
        return round(float(predicted_aqi), 2)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback: Calculate simple AQI
        if 'PM2.5' in current_data and 'PM10' in current_data:
            simple_aqi = 0.4 * current_data['PM2.5'] + 0.3 * current_data['PM10']
            return round(simple_aqi, 2)
        return 100.0  # Default

# Example usage
if __name__ == "__main__":
    # Test with sample data
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
