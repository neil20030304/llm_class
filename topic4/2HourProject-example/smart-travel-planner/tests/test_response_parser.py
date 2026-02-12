"""Tests for the agent response markdown parser."""

import pytest

from src.utils.response_parser import parse_agent_response, _extract_list_items


SAMPLE_AGENT_RESPONSE = """\
### 🌤️ Weather Summary
Paris will enjoy warm weather with highs around 27°C and lows near 18°C.
Expect mostly clear skies with a slight chance of rain on Wednesday.

### 🎒 Packing List
**Clothing:**
- Light cotton shirts
- Comfortable walking shoes
- Linen pants or shorts

**Accessories:**
- Sunglasses
- Sunscreen SPF 50
- Compact umbrella (just in case)

**Essentials:**
- Reusable water bottle
- Phone charger

### 🎯 Activity Ideas
- Stroll through Luxembourg Gardens
- Visit the Louvre on the rainy afternoon
- Explore Montmartre and Sacré-Cœur
- Evening Seine river cruise
- Café-hop in Le Marais

### ⚠️ Special Notes
Wednesday may bring light showers — plan indoor activities for the afternoon.
"""


class TestParseAgentResponse:
    """Tests for the top-level parse_agent_response function."""

    def test_extracts_summary(self):
        result = parse_agent_response(SAMPLE_AGENT_RESPONSE)
        assert "warm weather" in result["summary"]
        assert "27°C" in result["summary"]

    def test_extracts_packing_items(self):
        result = parse_agent_response(SAMPLE_AGENT_RESPONSE)
        assert len(result["packing"]) >= 6
        assert "Light cotton shirts" in result["packing"]
        assert "Compact umbrella (just in case)" in result["packing"]

    def test_extracts_activities(self):
        result = parse_agent_response(SAMPLE_AGENT_RESPONSE)
        assert len(result["activities"]) == 5
        assert any("Louvre" in a for a in result["activities"])

    def test_preserves_raw_markdown(self):
        result = parse_agent_response(SAMPLE_AGENT_RESPONSE)
        assert result["raw"] == SAMPLE_AGENT_RESPONSE

    def test_handles_empty_string(self):
        result = parse_agent_response("")
        assert result["summary"] == ""
        assert result["packing"] == []
        assert result["activities"] == []

    def test_unstructured_text_becomes_summary(self):
        text = "It looks like it'll be sunny all week. Pack light!"
        result = parse_agent_response(text)
        assert result["summary"] == text
        assert result["packing"] == []

    def test_strips_bold_markers_from_items(self):
        md = """### 🎒 Packing List\n- **Rain jacket** for afternoon showers\n"""
        result = parse_agent_response(md)
        assert "Rain jacket for afternoon showers" in result["packing"]


class TestExtractListItems:
    """Unit tests for the bullet-point extractor."""

    def test_dash_bullets(self):
        text = "- Apples\n- Bananas\n- Cherries"
        assert _extract_list_items(text) == ["Apples", "Bananas", "Cherries"]

    def test_asterisk_bullets(self):
        text = "* Alpha\n* Beta"
        assert _extract_list_items(text) == ["Alpha", "Beta"]

    def test_numbered_list(self):
        text = "1. First\n2. Second\n3) Third"
        assert _extract_list_items(text) == ["First", "Second", "Third"]

    def test_ignores_plain_lines(self):
        text = "Some prose here.\n- Actual item\nMore prose."
        assert _extract_list_items(text) == ["Actual item"]

    def test_empty_string(self):
        assert _extract_list_items("") == []
