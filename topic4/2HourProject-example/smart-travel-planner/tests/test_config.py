"""Unit tests for configuration utilities."""

import os
import pytest
from unittest.mock import patch

from src.utils.config import get_api_key, validate_config


class TestGetApiKey:
    """Tests for get_api_key function."""
    
    @patch.dict(os.environ, {"TEST_API_KEY": "secret123"})
    def test_returns_existing_key(self):
        """Test that existing env vars are returned."""
        result = get_api_key("TEST_API_KEY")
        assert result == "secret123"
    
    def test_returns_none_for_missing_key(self):
        """Test that missing env vars return None."""
        result = get_api_key("DEFINITELY_NOT_A_REAL_KEY_12345")
        assert result is None


class TestValidateConfig:
    """Tests for validate_config function."""
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-test",
        "OPENWEATHER_API_KEY": "test-weather-key"
    })
    def test_all_keys_present(self):
        """Test validation when all keys are present."""
        result = validate_config()
        
        assert result["OPENAI_API_KEY"] is True
        assert result["OPENWEATHER_API_KEY"] is True
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_missing_weather_key(self):
        """Test validation when weather key is missing."""
        # Clear the weather key if it exists
        os.environ.pop("OPENWEATHER_API_KEY", None)
        
        result = validate_config()
        
        assert result["OPENAI_API_KEY"] is True
        assert result["OPENWEATHER_API_KEY"] is False
