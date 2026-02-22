"""
DAY 2: Data Processing Module
Processes raw API data into features for ML model
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        print("✅ Data Processor Initialized")
    
    def process_raw_api_response(self, api_response):
        """
        Convert raw API response to clean DataFrame
        """
        try:
            # Extract pollutants
            pollutants = api_response['list'][0]['components']
            
            # Create DataFrame
            df = pd.DataFrame([pollutants])
            
            # Rename columns for consistency
            df = df.rename(columns={
                'pm2_5': 'PM2.5',
                'pm10': 'PM10',
                'no2': 'NO2',
                'o3': 'O3',
                'so2': 'SO2',
                'co': 'CO',
                'nh3': 'NH3',
                'no': 'NO'
            })
            
            # Add timestamp
            df['timestamp'] = datetime.fromtimestamp(api_response['list'][0]['dt'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
            
            print(f"✅ Processed data for {df['timestamp'].iloc[0]}")
            return df
            
        except Exception as e:
            print(f"❌ Error processing API data: {e}")
            return None
    
    def create_features(self, df):
        """
        Create additional features for ML model
        """
        features = df.copy()
        
        # 1. Calculate AQI (Simplified Indian AQI formula)
        # AQI = max of individual pollutant AQIs
        # For now, we'll use weighted average for simplicity
        if 'PM2.5' in features.columns and 'PM10' in features.columns:
            # Simple weighted AQI calculation
            pm25_weight = features['PM2.5'].iloc[0] * 0.4
            pm10_weight = features['PM10'].iloc[0] * 0.3
            o3_weight = features['O3'].iloc[0] * 0.15 if 'O3' in features.columns else 0
            no2_weight = features['NO2'].iloc[0] * 0.15 if 'NO2' in features.columns else 0
            
            features['AQI'] = pm25_weight + pm10_weight + o3_weight + no2_weight
        
        # 2. Pollutant ratios
        if 'PM10' in features.columns and features['PM10'].iloc[0] > 0:
            features['PM25_PM10_ratio'] = features['PM2.5'] / features['PM10']
        
        # 3. Time-based features
        features['is_rush_hour'] = features['hour'].apply(
            lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 20) else 0
        )
        features['is_night'] = features['hour'].apply(lambda x: 1 if 22 <= x or x <= 6 else 0)
        features['is_weekend'] = features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 4. Month and season
        features['month'] = features['timestamp'].dt.month
        features['season'] = features['month'].apply(
            lambda x: 'Winter' if x in [12, 1, 2] else
                     'Summer' if x in [3, 4, 5] else
                     'Monsoon' if x in [6, 7, 8, 9] else 'Post-Monsoon'
        )
        
        print(f"✅ Created {len(features.columns)} features")
        return features
    
    def save_to_csv(self, df, filename):
        """Save processed data to CSV"""
        try:
            df.to_csv(filename, index=False)
            print(f"💾 Saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False

# Quick test function
def test_processor():
    """Test the data processor"""
    print("🧪 Testing Data Processor...")
    processor = DataProcessor()
    
    # Use your actual API response structure
    sample_data = {
        "list": [{
            "dt": int(datetime.now().timestamp()),
            "components": {
                "co": 472.51, "no": 0.48, "no2": 3.41, "o3": 137.60,
                "so2": 4.98, "pm2_5": 106.55, "pm10": 150.64, "nh3": 11.23
            }
        }]
    }
    
    print("\n1. Processing raw API data...")
    processed = processor.process_raw_api_response(sample_data)
    
    if processed is not None:
        print("\n2. Creating features...")
        features = processor.create_features(processed)
        
        print("\n3. Displaying processed data:")
        print(features[['timestamp', 'PM2.5', 'PM10', 'AQI', 'hour', 'is_rush_hour']])
        
        print("\n4. Saving to CSV...")
        processor.save_to_csv(features, 'data/processed_sample.csv')
        
        print("\n🎉 Data Processor Test Complete!")
        return features
    else:
        print("❌ Failed to process data")
        return None

if __name__ == "__main__":
    test_processor()