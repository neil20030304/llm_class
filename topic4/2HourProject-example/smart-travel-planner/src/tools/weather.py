"""Weather forecast tool using OpenWeatherMap API."""

import os
from typing import Any

import requests
from langchain.tools import tool


BASE_URL = "https://api.openweathermap.org/data/2.5"


def _get_api_key() -> str | None:
    """Get OpenWeatherMap API key from environment."""
    return os.getenv("OPENWEATHER_API_KEY")


def _parse_forecast_data(data: dict[str, Any], units: str) -> str:
    """
    Parse raw API response into human-readable forecast.
    
    Args:
        data: Raw JSON response from OpenWeatherMap
        units: Temperature units ('metric' or 'imperial')
    
    Returns:
        Formatted weather forecast string.
    """
    city_name = data["city"]["name"]
    country = data["city"]["country"]
    
    # Group forecasts by day (API returns 3-hour intervals)
    daily_forecasts: dict[str, list] = {}
    for item in data["list"]:
        date = item["dt_txt"].split(" ")[0]
        if date not in daily_forecasts:
            daily_forecasts[date] = []
        daily_forecasts[date].append(item)
    
    # Format output
    unit_symbol = "°C" if units == "metric" else "°F"
    speed_unit = "m/s" if units == "metric" else "mph"
    
    lines = [f"🌍 Weather Forecast for {city_name}, {country}", "=" * 45, ""]
    
    for date, forecasts in list(daily_forecasts.items())[:5]:
        temps = [f["main"]["temp"] for f in forecasts]
        feels_like = [f["main"]["feels_like"] for f in forecasts]
        humidity = sum(f["main"]["humidity"] for f in forecasts) // len(forecasts)
        
        # Get midday conditions (most representative)
        midday = forecasts[len(forecasts) // 2]
        conditions = midday["weather"][0]["description"]
        wind_speed = midday["wind"]["speed"]
        
        # Rain probability (0-1 scale, convert to %)
        rain_prob = max(f.get("pop", 0) for f in forecasts) * 100
        
        lines.append(f"📅 {date}")
        lines.append(f"   🌡️  Temp: {min(temps):.1f} - {max(temps):.1f}{unit_symbol}")
        lines.append(f"   🤔 Feels like: {min(feels_like):.1f} - {max(feels_like):.1f}{unit_symbol}")
        lines.append(f"   ☁️  Conditions: {conditions.title()}")
        lines.append(f"   💧 Humidity: {humidity}%")
        lines.append(f"   🌧️  Rain chance: {rain_prob:.0f}%")
        lines.append(f"   💨 Wind: {wind_speed:.1f} {speed_unit}")
        lines.append("")
    
    return "\n".join(lines)


@tool
def get_weather_forecast(city: str, units: str = "metric") -> str:
    """
    Fetch 5-day weather forecast for a city using OpenWeatherMap API.
    
    Use this tool when you need to know the weather conditions for a destination.
    The forecast includes temperature, conditions, humidity, wind, and rain probability.
    
    Args:
        city: Name of the city (e.g., "Paris", "Tokyo", "New York", "London, UK")
        units: Temperature units - "metric" (Celsius) or "imperial" (Fahrenheit).
               Defaults to metric.
    
    Returns:
        Formatted 5-day weather forecast with daily breakdowns, or an error message
        if the city is not found or the API request fails.
    
    Examples:
        - get_weather_forecast("Paris") -> 5-day forecast in Celsius
        - get_weather_forecast("Miami", units="imperial") -> forecast in Fahrenheit
    """
    api_key = _get_api_key()
    
    if not api_key:
        return (
            "❌ Error: OpenWeatherMap API key not configured.\n"
            "Please set OPENWEATHER_API_KEY in your .env file.\n"
            "Get a free key at: https://openweathermap.org/api"
        )
    
    try:
        response = requests.get(
            f"{BASE_URL}/forecast",
            params={
                "q": city,
                "appid": api_key,
                "units": units,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        
        return _parse_forecast_data(data, units)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return (
                f"❌ City '{city}' not found.\n"
                "Tips: Try including the country code (e.g., 'London, UK') "
                "or check the spelling."
            )
        if e.response.status_code == 401:
            return (
                "❌ Invalid API key.\n"
                "Please check your OPENWEATHER_API_KEY in .env.\n"
                "Note: New keys take ~10 minutes to activate!"
            )
        return f"❌ API error: {e.response.status_code} - {e.response.reason}"
        
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Please try again."
        
    except requests.exceptions.ConnectionError:
        return "❌ Network error. Please check your internet connection."
        
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"
        
    except (KeyError, IndexError, TypeError) as e:
        return f"❌ Error parsing API response: {e}"
