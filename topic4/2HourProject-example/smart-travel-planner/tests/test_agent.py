"""Tests for the travel planning agent.

Unit tests run without API keys; integration tests (marked @pytest.mark.integration)
require real keys and make live API calls.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestTravelAgentCreation:
    """Tests for agent creation and configuration."""
    
    def test_create_agent_imports(self):
        """Test that agent module can be imported."""
        from src.agent.travel_agent import (
            create_travel_agent,
            SYSTEM_PROMPT,
            build_trip_message,
            is_agent_available,
        )

        assert callable(create_travel_agent)
        assert callable(build_trip_message)
        assert callable(is_agent_available)
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100
    
    def test_system_prompt_contains_guidelines(self):
        """Test that system prompt includes key guidelines."""
        from src.agent.travel_agent import SYSTEM_PROMPT
        
        # Check for packing guidelines
        assert "Cold" in SYSTEM_PROMPT or "cold" in SYSTEM_PROMPT
        assert "Rain" in SYSTEM_PROMPT or "rain" in SYSTEM_PROMPT
        assert "Sunny" in SYSTEM_PROMPT or "sunny" in SYSTEM_PROMPT
        
        # Check for activity recommendations
        assert "Museum" in SYSTEM_PROMPT or "museum" in SYSTEM_PROMPT
        assert "outdoor" in SYSTEM_PROMPT.lower()
    
    @patch('src.agent.travel_agent.ChatOpenAI')
    @patch('src.agent.travel_agent.create_react_agent')
    def test_create_agent_uses_correct_model(self, mock_create_agent, mock_llm):
        """Test that agent is created with specified model."""
        from src.agent.travel_agent import create_travel_agent
        
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_create_agent.return_value = MagicMock()
        
        create_travel_agent(model="gpt-4o")
        
        mock_llm.assert_called_once_with(model="gpt-4o", temperature=0.7)
        # Verify create_react_agent was called with prompt parameter
        mock_create_agent.assert_called_once()
    
    @patch('src.agent.travel_agent.ChatOpenAI')
    @patch('src.agent.travel_agent.create_react_agent')
    def test_create_agent_custom_temperature(self, mock_create_agent, mock_llm):
        """Test that custom temperature is passed to LLM."""
        from src.agent.travel_agent import create_travel_agent

        mock_llm.return_value = MagicMock()
        mock_create_agent.return_value = MagicMock()

        create_travel_agent(temperature=0.3)

        mock_llm.assert_called_once_with(model="gpt-4o-mini", temperature=0.3)


class TestBuildTripMessage:
    """Tests for the web-helper message builder."""

    def test_city_only(self):
        from src.agent.travel_agent import build_trip_message
        msg = build_trip_message("Paris")
        assert "Paris" in msg
        assert "Celsius" in msg
        assert "pack" in msg.lower()

    def test_includes_dates(self):
        from src.agent.travel_agent import build_trip_message
        msg = build_trip_message("Tokyo", "imperial", "2025-08-01", "2025-08-07")
        assert "2025-08-01" in msg
        assert "2025-08-07" in msg
        assert "Fahrenheit" in msg

    def test_start_date_only(self):
        from src.agent.travel_agent import build_trip_message
        msg = build_trip_message("London", start_date="2025-09-10")
        assert "2025-09-10" in msg
        assert "arriving" in msg.lower()

    def test_end_date_only(self):
        from src.agent.travel_agent import build_trip_message
        msg = build_trip_message("Sydney", end_date="2025-12-25")
        assert "2025-12-25" in msg


class TestIsAgentAvailable:
    """Tests for the API-key check."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-real-key"}, clear=False)
    def test_returns_true_with_valid_key(self):
        from src.agent.travel_agent import is_agent_available
        assert is_agent_available() is True

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_returns_false_when_empty(self):
        from src.agent.travel_agent import is_agent_available
        assert is_agent_available() is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-your-openai-key-here"}, clear=False)
    def test_returns_false_for_placeholder(self):
        from src.agent.travel_agent import is_agent_available
        assert is_agent_available() is False


@pytest.mark.integration
class TestTravelAgentIntegration:
    """
    Integration tests that require real API keys.
    
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """
    
    @pytest.fixture
    def agent(self):
        """Create a real agent instance."""
        from src.agent.travel_agent import create_travel_agent
        return create_travel_agent()
    
    def test_agent_handles_basic_query(self, agent):
        """Test that agent can process a basic travel query."""
        from src.agent.travel_agent import invoke_agent
        
        result = invoke_agent(agent, "What's the weather like in Tokyo?")
        
        # Should return a non-empty response
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_agent_provides_packing_advice(self, agent):
        """Test that agent responds meaningfully to a packing query."""
        from src.agent.travel_agent import invoke_agent

        # Be very explicit so the agent doesn't ask for clarification
        result = invoke_agent(
            agent,
            "I'm traveling to London from tomorrow for 5 days. "
            "Please use Celsius. What should I pack?"
        )

        # Agent should return *something* non-trivial
        assert isinstance(result, str)
        assert len(result) > 20
