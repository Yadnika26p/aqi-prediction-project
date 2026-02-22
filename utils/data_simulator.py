"""
DAY 2: Data Simulator for generating historical/training data
Since OpenWeatherMap free tier doesn't provide historical data,
we'll create realistic simulated data for ML training
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class DataSimulator:
    def __init__(self, base_data=None):
        """
        Initialize with base data (from your real API call)
        """
        if base_data is None:
            # Default Delhi base data
            self.base_data = {
                'PM2.5': 106.55,
                'PM10': 150.64,
                'O3': 137.60,
                'NO2': 3.41,
                'SO2': 4.98,
                'CO': 472.51,
                'NH3': 11.23
            }
        else:
            self.base_data = base_data
        
        print("✅ Data Simulator Initialized")
    
    def generate_historical_data(self, days=60, records_per_day=24):
        """
        Generate historical data for training ML model
        
        Args:
            days: Number of days of historical data
            records_per_day: Data points per day (hourly = 24)
        """
        print(f"📊 Generating {days} days of historical data...")
        
        all_data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for hour in range(records_per_day):
                record_time = current_date.replace(hour=hour)
                
                # Generate realistic variations
                data_point = self._generate_realistic_data_point(record_time)
                all_data.append(data_point)
            
            # Show progress
            if (day + 1) % 10 == 0:
                print(f"  Generated {day + 1}/{days} days...")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        print(f"✅ Generated {len(df)} data points")
        return df
    
    def _generate_realistic_data_point(self, timestamp):
        """
        Generate one data point with realistic patterns
        """
        hour = timestamp.hour
        month = timestamp.month
        day_of_week = timestamp.weekday()
        
        # Base multipliers for different times
        if 22 <= hour or hour <= 6:  # Night (10 PM - 6 AM)
            hour_factor = 0.7 + random.random() * 0.3
        elif 7 <= hour <= 10:  # Morning rush hour
            hour_factor = 1.2 + random.random() * 0.4
        elif 17 <= hour <= 20:  # Evening rush hour
            hour_factor = 1.3 + random.random() * 0.5
        else:  # Normal hours
            hour_factor = 0.9 + random.random() * 0.4
        
        # Weekend effect
        if day_of_week >= 5:  # Saturday or Sunday
            day_factor = 0.8  # Lower pollution on weekends
        else:
            day_factor = 1.0
        
        # Seasonal patterns (Delhi specific)
        if month in [10, 11, 12, 1]:  # Oct-Jan: High pollution (winter, Diwali, stubble burning)
            season_factor = 1.5 + random.random() * 0.5
        elif month in [5, 6]:  # May-June: Summer, moderate
            season_factor = 1.0 + random.random() * 0.3
        else:  # Other months
            season_factor = 0.8 + random.random() * 0.4
        
        # Generate data point with all factors
        data_point = {
            'timestamp': timestamp,
            'date': timestamp.date(),
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_rush_hour': 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0,
            'is_night': 1 if 22 <= hour or hour <= 6 else 0,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'season': self._get_season(month)
        }
        
        # Add pollutants with realistic variations
        for pollutant, base_value in self.base_data.items():
            # Each pollutant has different variation patterns
            if pollutant in ['PM2.5', 'PM10']:
                poll_factor = hour_factor * day_factor * season_factor * (0.8 + random.random() * 0.4)
            elif pollutant == 'O3':
                # Ozone peaks in afternoon
                if 12 <= hour <= 16:
                    poll_factor = 1.5 + random.random() * 0.5
                else:
                    poll_factor = 0.7 + random.random() * 0.4
            else:
                poll_factor = hour_factor * day_factor * (0.7 + random.random() * 0.6)
            
            data_point[pollutant] = base_value * poll_factor
        
        # Calculate AQI (simplified)
        data_point['AQI'] = self._calculate_aqi(data_point)
        
        return data_point
    
    def _get_season(self, month):
        """Get season name from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'
    
    def _calculate_aqi(self, data_point):
        """Simple AQI calculation"""
        # Weighted average of major pollutants
        weights = {
            'PM2.5': 0.35,
            'PM10': 0.25,
            'O3': 0.20,
            'NO2': 0.10,
            'SO2': 0.10
        }
        
        aqi = 0
        for pollutant, weight in weights.items():
            if pollutant in data_point:
                aqi += data_point[pollutant] * weight
        
        return aqi
    
    def save_to_csv(self, df, filename='data/training_dataset.csv'):
        """Save generated data to CSV"""
        try:
            df.to_csv(filename, index=False)
            print(f"💾 Training dataset saved to: {filename}")
            print(f"   Records: {len(df)}")
            print(f"   Columns: {len(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False

# Test function
def test_simulator():
    """Test the data simulator"""
    print("🧪 Testing Data Simulator...")
    
    # Use your real API data as base
    base_data = {
        'PM2.5': 106.55,
        'PM10': 150.64,
        'O3': 137.60,
        'NO2': 3.41,
        'SO2': 4.98,
        'CO': 472.51,
        'NH3': 11.23
    }
    
    simulator = DataSimulator(base_data)
    
    # Generate 7 days of data for testing (24 hours each)
    print("\nGenerating test dataset...")
    df = simulator.generate_historical_data(days=7, records_per_day=24)
    
    print("\n📊 Sample of generated data:")
    print(df[['timestamp', 'PM2.5', 'PM10', 'AQI', 'hour', 'is_rush_hour']].head())
    
    print("\n📈 Statistics:")
    print(f"PM2.5 - Avg: {df['PM2.5'].mean():.2f}, Min: {df['PM2.5'].min():.2f}, Max: {df['PM2.5'].max():.2f}")
    print(f"PM10 - Avg: {df['PM10'].mean():.2f}, Min: {df['PM10'].min():.2f}, Max: {df['PM10'].max():.2f}")
    print(f"AQI - Avg: {df['AQI'].mean():.2f}, Min: {df['AQI'].min():.2f}, Max: {df['AQI'].max():.2f}")
    
    # Save the data
    simulator.save_to_csv(df, 'data/training_sample.csv')
    
    return df

if __name__ == "__main__":
    test_simulator()