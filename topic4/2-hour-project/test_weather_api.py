"""
Simple test script to verify OpenWeatherMap API is working
Uses the One Call API 3.0 with lat/lon coordinates
"""

import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Get API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"
BASE_URL = "https://api.openweathermap.org/data/2.5"
print("=" * 60)
print("OpenWeatherMap One Call API 3.0 Test")
print("=" * 60)

# Check if API key exists
if not OPENWEATHER_API_KEY:
    print("❌ Error: OPENWEATHER_API_KEY not found in .env file")
    exit(1)

print(f"\n✓ API Key found: {OPENWEATHER_API_KEY[:10]}..." )

# Test parameters
test_city = "London"
units = "metric"

print(f"✓ Testing with city: {test_city}")

# Step 1: Geocode the city to get lat/lon
print(f"\n📡 Step 1: Geocoding city '{test_city}'...")
print(f"URL: {GEO_URL}")

try:
    geo_response = requests.get(
        GEO_URL,
        params={
            "q": test_city,
            "limit": 1,
            "appid": OPENWEATHER_API_KEY
        },
        timeout=10
    )
    
    print(f"📊 Geocoding Status Code: {geo_response.status_code}")
    
    if geo_response.status_code != 200:
        print(f"❌ FAILED: Geocoding failed with status {geo_response.status_code}")
        print(f"Response: {geo_response.text[:200]}")
        exit(1)
    
    geo_data = geo_response.json()
    
    if not geo_data:
        print(f"❌ FAILED: City '{test_city}' not found")
        exit(1)
    
    lat = geo_data[0]["lat"]
    lon = geo_data[0]["lon"]
    city_name = geo_data[0]["name"]
    country = geo_data[0].get("country", "")
    
    print(f"✅ Found: {city_name}, {country}")
    print(f"   Latitude: {lat}")
    print(f"   Longitude: {lon}")
    
    # Step 2: Call One Call API 3.0
    print(f"\n📡 Step 2: Fetching weather data...")
    print(f"URL: {ONECALL_URL}")
    print(f"Parameters: lat={lat}, lon={lon}, units={units}")
    
    weather_response = requests.get(
        ONECALL_URL,
        params={
            "lat": lat,
            "lon": lon,
            "exclude": "minutely,alerts",  # Exclude minutely data and alerts
            "units": units,
            "appid": OPENWEATHER_API_KEY
        },
        timeout=10
    )
    
    print(f"\n📊 Weather API Status Code: {weather_response.status_code}")
    
    # Check if request was successful
    if weather_response.status_code == 200:
        print("✅ SUCCESS! API call worked.\n")
        
        # Parse JSON
        data = weather_response.json()
        
        # Print formatted response
        print("=" * 60)
        print("API RESPONSE DATA:")
        print("=" * 60)
        
        # Location info
        print(f"\n🌍 Location: {city_name}, {country}")
        print(f"📍 Coordinates: lat={data['lat']}, lon={data['lon']}")
        print(f"🕐 Timezone: {data.get('timezone', 'N/A')}")
        
        # Current weather
        if data.get('current'):
            current = data['current']
            print(f"\n🌤️  Current Weather:")
            print(f"   Temperature: {current['temp']}°C")
            print(f"   Feels Like: {current['feels_like']}°C")
            print(f"   Conditions: {current['weather'][0]['description']}")
            print(f"   Humidity: {current['humidity']}%")
            print(f"   Wind Speed: {current['wind_speed']} m/s")
            print(f"   UV Index: {current.get('uvi', 'N/A')}")
        
        # Daily forecast (first 3 days)
        if data.get('daily'):
            print(f"\n📅 Daily Forecast (next 3 days):")
            for i, day in enumerate(data['daily'][:3]):
                from datetime import datetime
                date = datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d')
                print(f"\n   {date}:")
                print(f"      Temp: {day['temp']['min']:.1f} - {day['temp']['max']:.1f}°C")
                print(f"      Conditions: {day['weather'][0]['description']}")
                print(f"      Rain Chance: {day.get('pop', 0) * 100:.0f}%")
                print(f"      Humidity: {day['humidity']}%")
        
        # Print sample JSON
        print("\n" + "=" * 60)
        print("SAMPLE JSON RESPONSE (current + first daily):")
        print("=" * 60)
        
        sample_data = {
            "lat": data["lat"],
            "lon": data["lon"],
            "timezone": data.get("timezone"),
            "current": data.get("current"),
            "daily": data.get("daily", [])[:1]  # Just first day
        }
        print(json.dumps(sample_data, indent=2, default=str))
        
    elif weather_response.status_code == 401:
        print("❌ FAILED: Invalid API Key (401 Unauthorized)")
        print(f"\nYour API key: {OPENWEATHER_API_KEY}")
        print("\nPossible issues:")
        print("  1. API key is incorrect")
        print("  2. API key is new and not activated yet (wait 10-15 minutes)")
        print("  3. API key was revoked or expired")
        print("  4. One Call API 3.0 requires a subscription (free tier may not include it)")
        print("\nNote: One Call API 3.0 may require subscribing at:")
        print("  https://openweathermap.org/api/one-call-3")
        print("\nGenerate a new key at: https://home.openweathermap.org/api_keys")
        
    elif weather_response.status_code == 404:
        print("❌ FAILED: Endpoint not found (404)")
        print("The API endpoint may have changed.")
        
    elif weather_response.status_code == 429:
        print("❌ FAILED: Rate limit exceeded (429)")
        print("You've made too many requests. Wait a minute and try again.")
        
    else:
        print(f"❌ FAILED: Unexpected status code {weather_response.status_code}")
        print(f"Response: {weather_response.text[:500]}")
    
except requests.exceptions.Timeout:
    print("❌ FAILED: Request timeout")
    print("The API server took too long to respond. Check your internet connection.")
    
except requests.exceptions.ConnectionError as e:
    print("❌ FAILED: Connection error")
    print(f"Could not connect to OpenWeatherMap API: {e}")
    print("\nCheck your internet connection.")
    
except Exception as e:
    print(f"❌ FAILED: Unexpected error")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)

