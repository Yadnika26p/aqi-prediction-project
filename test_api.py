"""
DAY 2: Enhanced API Test with Data Processing
"""
import requests
import sys
from datetime import datetime
import pandas as pd
from data_processor import DataProcessor

print("=" * 50)
print("🌍 AQI PROJECT - DAY 2: API + Data Processing")
print("=" * 50)

# Try to import config
try:
    from config import OPENWEATHER_API_KEY, DELHI_COORDS
    print("✅ Config loaded successfully")
except ImportError as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

# Initialize data processor
processor = DataProcessor()

# Build API URL for Delhi
lat = DELHI_COORDS["lat"]
lon = DELHI_COORDS["lon"]
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"

print(f"\n📍 Fetching LIVE data for: {DELHI_COORDS['name']}")

# Try to fetch from API
print("\n" + "="*30)
print("🔄 Calling OpenWeatherMap API...")
print("="*30)

try:
    response = requests.get(url, timeout=10)
    print(f"📊 HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("🎉 SUCCESS! Real API data received!")
        
        # Process the data
        print("\n" + "="*30)
        print("⚙️ Processing API Data...")
        print("="*30)
        
        processed_data = processor.process_raw_api_response(data)
        
        if processed_data is not None:
            # Create features
            features = processor.create_features(processed_data)
            
            # Display results
            print("\n" + "="*50)
            print("📈 PROCESSED AIR QUALITY DATA")
            print("="*50)
            
            # Show key metrics
            print(f"📍 Location: {DELHI_COORDS['name']}")
            print(f"🕐 Time: {features['timestamp'].iloc[0]}")
            print(f"📅 Day: {features['day_of_week'].iloc[0]} (0=Monday, 6=Sunday)")
            print(f"🌡️ Rush Hour: {'Yes' if features['is_rush_hour'].iloc[0] else 'No'}")
            print(f"🌙 Night Time: {'Yes' if features['is_night'].iloc[0] else 'No'}")
            print(f"🎉 Weekend: {'Yes' if features['is_weekend'].iloc[0] else 'No'}")
            
            print("\n🔬 POLLUTANT LEVELS:")
            print("-" * 50)
            pollutants = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
            for poll in pollutants:
                if poll in features.columns:
                    value = features[poll].iloc[0]
                    unit = "μg/m³" if poll != 'CO' else "μg/m³"
                    print(f"  {poll:<6}: {value:>8.2f} {unit}")
            
            print("\n📊 CALCULATED AQI:")
            print("-" * 50)
            if 'AQI' in features.columns:
                aqi = features['AQI'].iloc[0]
                print(f"  Estimated AQI: {aqi:.2f}")
                
                # Health advisory
                if aqi <= 50:
                    advisory = "✅ Good - Air quality is satisfactory"
                elif aqi <= 100:
                    advisory = "⚠️ Moderate - Acceptable air quality"
                elif aqi <= 150:
                    advisory = "😷 Poor - Unhealthy for sensitive groups"
                elif aqi <= 200:
                    advisory = "🚨 Very Poor - Everyone may experience health effects"
                elif aqi <= 300:
                    advisory = "🆘 Severe - Health alert"
                else:
                    advisory = "☠️ Hazardous - Emergency conditions"
                
                print(f"  Health Advisory: {advisory}")
            
            # Save the processed data
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/delhi_processed_{timestamp_str}.csv"
            processor.save_to_csv(features, filename)
            
            # Also append to main dataset
            main_file = "data/delhi_historical.csv"
            try:
                # Try to read existing data
                historical_df = pd.read_csv(main_file)
                combined_df = pd.concat([historical_df, features], ignore_index=True)
                combined_df.to_csv(main_file, index=False)
                print(f"💾 Appended to main dataset: {main_file}")
            except FileNotFoundError:
                # First time - create new file
                features.to_csv(main_file, index=False)
                print(f"💾 Created new dataset: {main_file}")
            
        else:
            print("❌ Failed to process API data")
            
    else:
        print(f"❌ API Error: {response.status_code}")
        print("Response:", response.text[:200])
        
except requests.exceptions.Timeout:
    print("⏰ Timeout: API took too long to respond")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50)
print("✅ DAY 2 COMPLETE! Real data processed successfully!")
print("="*50)