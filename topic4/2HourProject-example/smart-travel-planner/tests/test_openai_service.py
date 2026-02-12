"""Tests for the OpenAI travel recommendation service."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.services.openai_service import generate_travel_recommendations


SAMPLE_WEATHER = """
🌍 Weather Forecast for Paris, FR
=============================================

📅 2025-07-20
   🌡️  Temp: 22.0 - 28.5°C
   🤔 Feels like: 21.5 - 29.0°C
   ☁️  Conditions: Clear Sky
   💧 Humidity: 45%
   🌧️  Rain chance: 10%
   💨 Wind: 3.2 m/s
"""

SAMPLE_AI_RESPONSE = json.dumps({
    "summary": "Expect warm and sunny weather in Paris with highs around 28°C.",
    "packing": [
        "Light cotton shirts",
        "Comfortable walking shoes",
        "Sunscreen SPF 50",
        "Sunglasses",
        "Light cardigan for evenings",
        "Reusable water bottle",
        "Crossbody bag for sightseeing",
        "Linen pants or shorts",
    ],
    "activities": [
        "Stroll along the Seine at sunset",
        "Picnic at Luxembourg Gardens",
        "Visit the Louvre (air-conditioned!)",
        "Explore Montmartre and Sacré-Cœur",
        "Evening rooftop drinks near the Eiffel Tower",
        "Browse the Marais flea markets",
    ],
})


class TestGetClient:
    """Tests for _get_client helper."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_returns_none_when_key_empty(self):
        from src.services.openai_service import _get_client
        assert _get_client() is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-your-openai-key-here"}, clear=False)
    def test_returns_none_for_placeholder_key(self):
        from src.services.openai_service import _get_client
        assert _get_client() is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-real-key-abc123"}, clear=False)
    @patch("src.services.openai_service.OpenAI")
    def test_returns_client_with_valid_key(self, mock_openai_cls):
        from src.services.openai_service import _get_client
        client = _get_client()
        assert client is not None
        mock_openai_cls.assert_called_once_with(api_key="sk-real-key-abc123")


class TestGenerateRecommendations:
    """Tests for generate_travel_recommendations."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_returns_none_when_no_api_key(self):
        result = generate_travel_recommendations("Paris", SAMPLE_WEATHER)
        assert result is None

    @patch("src.services.openai_service._get_client")
    def test_returns_structured_data_on_success(self, mock_get_client):
        # Mock the OpenAI response chain
        mock_message = MagicMock()
        mock_message.content = SAMPLE_AI_RESPONSE

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_travel_recommendations("Paris", SAMPLE_WEATHER)

        assert result is not None
        assert "summary" in result
        assert isinstance(result["packing"], list)
        assert isinstance(result["activities"], list)
        assert len(result["packing"]) > 0
        assert len(result["activities"]) > 0

    @patch("src.services.openai_service._get_client")
    def test_returns_none_on_invalid_json(self, mock_get_client):
        mock_message = MagicMock()
        mock_message.content = "not valid json at all"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_travel_recommendations("Paris", SAMPLE_WEATHER)
        assert result is None

    @patch("src.services.openai_service._get_client")
    def test_returns_none_on_missing_keys(self, mock_get_client):
        # Response is valid JSON but missing required keys
        mock_message = MagicMock()
        mock_message.content = json.dumps({"foo": "bar"})

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_travel_recommendations("Paris", SAMPLE_WEATHER)
        assert result is None
