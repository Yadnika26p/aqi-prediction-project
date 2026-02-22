 
import os
from dotenv import load_dotenv
import sys

# For Windows path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv()

# API Configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Print for debugging (Windows often has path issues)
print(f"📁 Current directory: {BASE_DIR}")
print(f"🔑 API Key loaded: {'Yes' if OPENWEATHER_API_KEY else 'No'}")

# City coordinates (Delhi)
DELHI_COORDS = {
    "lat": 28.6139,
    "lon": 77.2090,
    "name": "Delhi"
}

# API Endpoints
AIR_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"