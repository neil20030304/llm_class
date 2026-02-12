"""Unit tests for the weather forecast tool."""

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.tools.weather import get_weather_forecast, _parse_forecast_data


# Sample API response for testing
SAMPLE_API_RESPONSE = {
    "cod": "200",
    "city": {
        "name": "Paris",
        "country": "FR"
    },
    "list": [
        {
            "dt": 1718438400,
            "dt_txt": "2024-06-15 12:00:00",
            "main": {
                "temp": 22.5,
                "feels_like": 21.8,
                "temp_min": 20.1,
                "temp_max": 24.3,
                "humidity": 65
            },
            "weather": [
                {
                    "main": "Clouds",
                    "description": "scattered clouds",
                    "icon": "03d"
                }
            ],
            "wind": {
                "speed": 3.5
            },
            "pop": 0.2
        },
        {
            "dt": 1718449200,
            "dt_txt": "2024-06-15 15:00:00",
            "main": {
                "temp": 24.0,
                "feels_like": 23.5,
                "temp_min": 22.0,
                "temp_max": 25.0,
                "humidity": 60
            },
            "weather": [
                {
                    "main": "Clear",
                    "description": "clear sky",
                    "icon": "01d"
                }
            ],
            "wind": {
                "speed": 4.0
            },
            "pop": 0.1
        }
    ]
}


class TestParserFunction:
    """Tests for the forecast data parser."""
    
    def test_parse_forecast_data_metric(self):
        """Test parsing with metric units."""
        result = _parse_forecast_data(SAMPLE_API_RESPONSE, "metric")
        
        assert "Paris" in result
        assert "FR" in result
        assert "°C" in result
        assert "m/s" in result
        assert "2024-06-15" in result
    
    def test_parse_forecast_data_imperial(self):
        """Test parsing with imperial units."""
        result = _parse_forecast_data(SAMPLE_API_RESPONSE, "imperial")
        
        assert "Paris" in result
        assert "°F" in result
        assert "mph" in result


class TestWeatherTool:
    """Tests for the weather forecast tool."""
    
    @patch('src.tools.weather._get_api_key')
    def test_missing_api_key(self, mock_get_key):
        """Test error handling when API key is missing."""
        mock_get_key.return_value = None
        
        result = get_weather_forecast.invoke({"city": "Paris"})
        
        assert "API key not configured" in result
        assert "OPENWEATHER_API_KEY" in result
    
    @patch('src.tools.weather.requests.get')
    @patch('src.tools.weather._get_api_key')
    def test_successful_forecast(self, mock_get_key, mock_get):
        """Test successful weather retrieval."""
        mock_get_key.return_value = "test-api-key"
        
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = get_weather_forecast.invoke({"city": "Paris"})
        
        assert "Paris" in result
        assert "FR" in result
        assert "°C" in result
    
    @patch('src.tools.weather.requests.get')
    @patch('src.tools.weather._get_api_key')
    def test_city_not_found(self, mock_get_key, mock_get):
        """Test handling of invalid city name."""
        mock_get_key.return_value = "test-api-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response
        
        result = get_weather_forecast.invoke({"city": "NotARealCity123"})
        
        assert "not found" in result.lower()
    
    @patch('src.tools.weather.requests.get')
    @patch('src.tools.weather._get_api_key')
    def test_invalid_api_key(self, mock_get_key, mock_get):
        """Test handling of invalid API key."""
        mock_get_key.return_value = "bad-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response
        
        result = get_weather_forecast.invoke({"city": "Paris"})
        
        assert "Invalid API key" in result
    
    @patch('src.tools.weather.requests.get')
    @patch('src.tools.weather._get_api_key')
    def test_timeout_handling(self, mock_get_key, mock_get):
        """Test handling of request timeout."""
        mock_get_key.return_value = "test-api-key"
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = get_weather_forecast.invoke({"city": "Paris"})
        
        assert "timed out" in result.lower()
    
    @patch('src.tools.weather.requests.get')
    @patch('src.tools.weather._get_api_key')
    def test_network_error(self, mock_get_key, mock_get):
        """Test handling of network errors."""
        mock_get_key.return_value = "test-api-key"
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = get_weather_forecast.invoke({"city": "Paris"})
        
        assert "Network error" in result
    
    @patch('src.tools.weather.requests.get')
    @patch('src.tools.weather._get_api_key')
    def test_imperial_units(self, mock_get_key, mock_get):
        """Test using imperial units."""
        mock_get_key.return_value = "test-api-key"
        
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = get_weather_forecast.invoke({
            "city": "Miami",
            "units": "imperial"
        })
        
        # Verify the API was called with imperial units
        call_args = mock_get.call_args
        assert call_args[1]["params"]["units"] == "imperial"
