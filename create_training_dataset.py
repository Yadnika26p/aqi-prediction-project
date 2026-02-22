"""
DAY 2: Create complete training dataset
Combines real API data with simulated historical data
"""
import pandas as pd
from utils.data_simulator import DataSimulator
from data_processor import DataProcessor
import requests
from config import OPENWEATHER_API_KEY, DELHI_COORDS
from datetime import datetime

def fetch_current_real_data():
    """Fetch current real data from API"""
    print("🌍 Fetching current real data...")
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={DELHI_COORDS['lat']}&lon={DELHI_COORDS['lon']}&appid={OPENWEATHER_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Process the data
            processor = DataProcessor()
            processed = processor.process_raw_api_response(data)
            
            if processed is not None:
                features = processor.create_features(processed)
                print(f"✅ Fetched real data for {features['timestamp'].iloc[0]}")
                return features.iloc[0].to_dict()  # Return as dict
        else:
            print(f"❌ API Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error fetching real data: {e}")
    
    return None

def create_complete_dataset():
    """Create complete dataset for ML training"""
    print("=" * 60)
    print("🧠 CREATING COMPLETE TRAINING DATASET")
    print("=" * 60)
    
    # Step 1: Get current real data as base
    real_data = fetch_current_real_data()
    
    if real_data:
        # Extract base pollutant levels from real data
        base_data = {}
        pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 'NH3']
        for poll in pollutants:
            if poll in real_data:
                base_data[poll] = real_data[poll]
        
        print(f"\n📊 Using real data as base:")
        for poll, value in base_data.items():
            print(f"  {poll}: {value:.2f}")
    else:
        # Use default values if real data not available
        base_data = None
        print("⚠️ Using default base data")
    
    # Step 2: Generate historical data
    print("\n" + "="*30)
    print("⚙️ Generating Historical Data")
    print("="*30)
    
    simulator = DataSimulator(base_data)
    
    # Generate 60 days of data (for good ML training)
    # This is simulated but realistic based on patterns
    historical_df = simulator.generate_historical_data(days=60, records_per_day=24)
    
    # Step 3: Save the complete dataset
    print("\n" + "="*30)
    print("💾 Saving Complete Dataset")
    print("="*30)
    
    output_file = "data/complete_training_dataset.csv"
    historical_df.to_csv(output_file, index=False)
    
    print(f"\n✅ DATASET CREATION COMPLETE!")
    print(f"📁 File: {output_file}")
    print(f"📊 Records: {len(historical_df):,}")
    print(f"📈 Columns: {len(historical_df.columns)}")
    
    # Show sample
    print(f"\n📋 Sample data:")
    print(historical_df[['timestamp', 'PM2.5', 'PM10', 'AQI', 'season', 'is_rush_hour']].head())
    
    # Show statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Time range: {historical_df['timestamp'].min()} to {historical_df['timestamp'].max()}")
    print(f"   PM2.5 range: {historical_df['PM2.5'].min():.1f} to {historical_df['PM2.5'].max():.1f} μg/m³")
    print(f"   AQI range: {historical_df['AQI'].min():.1f} to {historical_df['AQI'].max():.1f}")
    
    return historical_df

if __name__ == "__main__":
    dataset = create_complete_dataset()
    
    print("\n" + "="*60)
    print("🎯 READY FOR DAY 3: MACHINE LEARNING MODEL!")
    print("="*60)
    print("\nNext steps:")
    print("1. We have training data ready")
    print("2. Day 3: Build ML model to predict AQI")
    print("3. You can explore the dataset in data/complete_training_dataset.csv")