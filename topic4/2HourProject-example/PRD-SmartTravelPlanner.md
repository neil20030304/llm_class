# Product Requirements Document (PRD)
## Smart Travel Planner Agent
### Using OpenWeatherMap API + LangChain ReAct Agent

---

## 📋 Document Information

| Field | Value |
|-------|-------|
| **Project Name** | Smart Travel Planner |
| **Version** | 1.0 |
| **Author** | CS6501 - Topic 4: Exploring Tools |
| **Date** | 2025 |
| **Timeline** | 2 Hours |
| **Status** | Draft |

---

## 1. Executive Summary

The **Smart Travel Planner** is an AI-powered agent that helps users prepare for trips by analyzing weather forecasts for their destination. Given a city and travel dates, the agent fetches real-time weather data from OpenWeatherMap, then intelligently generates personalized packing lists and activity recommendations based on predicted conditions.

This project serves as an educational exercise in building LLM-powered agents with external API tool integration, emphasizing real-world patterns like API authentication, structured data handling, and conditional reasoning.

---

## 2. Problem Statement

### The Pain Point
Travelers often struggle with:
- **Packing anxiety**: Not knowing what clothes/gear to bring for unpredictable weather
- **Activity planning**: Missing out on weather-dependent activities (beach days, hiking, etc.)
- **Information overload**: Having to manually check forecasts and mentally translate them into actionable decisions
- **Last-minute surprises**: Arriving at a destination unprepared for conditions

### The Opportunity
An intelligent agent can bridge the gap between raw weather data and actionable travel preparation advice, saving time and reducing travel stress.

---

## 3. Goals & Objectives

### Primary Goals
1. **Functional**: Build a working travel planning agent that fetches weather and provides useful recommendations
2. **Educational**: Teach API authentication patterns and structured data handling
3. **Practical**: Create something students might actually use in real life

### Learning Outcomes
| Outcome | Description |
|---------|-------------|
| API Authentication | Understanding API keys, environment variables, secure credential handling |
| Structured Data Handling | Parsing JSON responses, extracting relevant fields |
| Conditional Reasoning | Agent makes decisions based on external data (weather → recommendations) |
| Tool Integration | Connecting external services to LLM agents |

### Success Criteria
- [ ] Agent successfully retrieves weather data for any valid city
- [ ] Agent generates relevant packing recommendations based on temperature, precipitation, and conditions
- [ ] Agent suggests appropriate activities for the forecasted weather
- [ ] Handles edge cases gracefully (invalid city, API errors, etc.)

---

## 4. User Stories

### US-001: Basic Trip Planning
> **As a** traveler  
> **I want to** enter my destination city and travel dates  
> **So that** I receive weather-appropriate packing suggestions and activity ideas

**Acceptance Criteria:**
- User can input any major city name
- User can specify travel dates (or default to next 5 days)
- System returns temperature range, conditions summary, packing list, and activities

---

### US-002: Multi-Day Trip Analysis
> **As a** traveler planning a week-long trip  
> **I want to** see a breakdown of weather across my travel dates  
> **So that** I can pack for varying conditions throughout my stay

**Acceptance Criteria:**
- Agent provides day-by-day weather summary
- Packing list accounts for the full range of conditions
- Highlights any extreme weather days

---

### US-003: Activity Recommendations
> **As a** traveler unfamiliar with my destination  
> **I want to** receive activity suggestions based on weather  
> **So that** I can plan weather-appropriate excursions

**Acceptance Criteria:**
- Rainy days → indoor activities (museums, cafes, shopping)
- Sunny days → outdoor activities (parks, beaches, hiking)
- Cold days → cozy activities (hot springs, warm restaurants)
- Hot days → water activities, air-conditioned venues

---

## 5. Functional Requirements

### FR-001: Weather Data Retrieval
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-001.1 | Fetch current weather for specified city | P0 (Must Have) |
| FR-001.2 | Fetch 5-day forecast for trip planning | P0 (Must Have) |
| FR-001.3 | Parse temperature (min/max/feels-like) | P0 (Must Have) |
| FR-001.4 | Parse precipitation probability | P1 (Should Have) |
| FR-001.5 | Parse wind speed and humidity | P2 (Nice to Have) |

### FR-002: Packing List Generation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-002.1 | Generate clothing recommendations based on temperature | P0 |
| FR-002.2 | Suggest rain gear when precipitation > 30% | P0 |
| FR-002.3 | Include sun protection for clear/sunny days | P1 |
| FR-002.4 | Recommend layers for large temperature swings | P1 |
| FR-002.5 | Suggest specific items (umbrella, sunscreen, jacket type) | P2 |

### FR-003: Activity Recommendations
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-003.1 | Suggest indoor activities for bad weather | P1 |
| FR-003.2 | Suggest outdoor activities for good weather | P1 |
| FR-003.3 | Provide temperature-appropriate activity types | P2 |
| FR-003.4 | Warn about extreme weather conditions | P2 |

### FR-004: Error Handling
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-004.1 | Handle invalid city names gracefully | P0 |
| FR-004.2 | Handle API rate limiting | P1 |
| FR-004.3 | Handle network failures | P1 |
| FR-004.4 | Provide helpful error messages to user | P1 |

---

## 6. Technical Requirements

### 6.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|----------|
| **Language** | Python 3.10+ | Industry standard for AI/ML |
| **LLM Framework** | LangChain | Course standard, great tool abstraction |
| **Agent Type** | ReAct (via LangGraph) | Reasoning + Acting pattern |
| **LLM Provider** | OpenAI GPT-4o-mini | Cost-effective, capable |
| **Weather API** | OpenWeatherMap | Free tier, well-documented |
| **HTTP Client** | `requests` | Simple, reliable |
| **Config** | `python-dotenv` | Secure credential management |

### 6.2 Dependencies

```txt
# requirements.txt
langchain>=0.1.0
langchain-openai>=0.0.5
langgraph>=0.0.20
requests>=2.31.0
python-dotenv>=1.0.0
```

### 6.3 Environment Variables

```bash
# .env file (DO NOT COMMIT)
OPENAI_API_KEY=sk-your-openai-key-here
OPENWEATHER_API_KEY=your-openweathermap-key-here
```

### 6.4 API Configuration

**OpenWeatherMap API Details:**

| Setting | Value |
|---------|-------|
| Base URL | `https://api.openweathermap.org/data/2.5` |
| Forecast Endpoint | `/forecast` |
| Current Weather Endpoint | `/weather` |
| Free Tier Limits | 60 calls/minute, 1,000,000 calls/month |
| Response Format | JSON |
| Units | Metric (Celsius) or Imperial (Fahrenheit) |

---

## 7. System Architecture

### 7.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│        "Plan my trip to Paris from June 15-20"                  │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ReAct AGENT (LangGraph)                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   REASON    │───▶│     ACT     │───▶│   OBSERVE   │──┐      │
│  │ (Think)     │    │ (Use Tool)  │    │ (Get Result)│  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘  │      │
│         ▲                                               │      │
│         └───────────────────────────────────────────────┘      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WEATHER TOOL                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  get_weather_forecast(city: str) -> WeatherData     │       │
│  └─────────────────────────────────┬───────────────────┘       │
└────────────────────────────────────┼────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OpenWeatherMap API                              │
│           api.openweathermap.org/data/2.5/forecast              │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Data Flow

```
1. User Input: "Plan trip to Tokyo, May 10-15"
           │
           ▼
2. Agent Reasoning: "I need weather data for Tokyo"
           │
           ▼
3. Tool Invocation: get_weather_forecast("Tokyo")
           │
           ▼
4. API Call: GET /forecast?q=Tokyo&appid=XXX&units=metric
           │
           ▼
5. JSON Response: { temp: 22°C, rain: 30%, conditions: "cloudy" }
           │
           ▼
6. Agent Analysis: "Mild temps, chance of rain"
           │
           ▼
7. Output Generation:
   - Packing: Light jacket, umbrella, layers
   - Activities: Mix of indoor/outdoor plans
```

---

## 8. API Response Schema

### OpenWeatherMap 5-Day Forecast Response (Simplified)

```json
{
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
    }
  ]
}
```

### Key Fields to Extract

| Field | Path | Use Case |
|-------|------|----------|
| City Name | `city.name` | Confirm correct location |
| Temperature | `list[].main.temp` | Clothing recommendations |
| Feels Like | `list[].main.feels_like` | Comfort-based suggestions |
| Humidity | `list[].main.humidity` | Breathable fabric recommendations |
| Conditions | `list[].weather[0].main` | Activity planning |
| Description | `list[].weather[0].description` | Detailed condition info |
| Rain Probability | `list[].pop` | Rain gear recommendations |
| Wind Speed | `list[].wind.speed` | Windbreaker suggestions |

---

## 9. Implementation Guide

### 9.1 Project Structure

```
smart-travel-planner/
├── .env                    # API keys (gitignored)
├── .env.example            # Template for .env
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py             # Entry point
│   ├── tools/
│   │   ├── __init__.py
│   │   └── weather.py      # Weather tool definition
│   ├── agent/
│   │   ├── __init__.py
│   │   └── travel_agent.py # Agent configuration
│   └── utils/
│       ├── __init__.py
│       └── config.py       # Environment loading
└── tests/
    ├── __init__.py
    ├── test_weather_tool.py
    └── test_agent.py
```

### 9.2 Core Implementation

#### `src/tools/weather.py`

```python
import os
import requests
from langchain.tools import tool
from typing import Optional

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5"


@tool
def get_weather_forecast(city: str, units: str = "metric") -> str:
    """
    Fetch 5-day weather forecast for a city.
    
    Args:
        city: Name of the city (e.g., "Paris", "Tokyo", "New York")
        units: Temperature units - "metric" (Celsius) or "imperial" (Fahrenheit)
    
    Returns:
        Formatted weather forecast including temperature, conditions, 
        and precipitation probability for the next 5 days.
    """
    if not OPENWEATHER_API_KEY:
        return "Error: OpenWeatherMap API key not configured. Set OPENWEATHER_API_KEY."
    
    try:
        response = requests.get(
            f"{BASE_URL}/forecast",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": units
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Parse and format the forecast
        city_name = data["city"]["name"]
        country = data["city"]["country"]
        
        # Group forecasts by day (API returns 3-hour intervals)
        daily_forecasts = {}
        for item in data["list"]:
            date = item["dt_txt"].split(" ")[0]
            if date not in daily_forecasts:
                daily_forecasts[date] = []
            daily_forecasts[date].append(item)
        
        # Format output
        unit_symbol = "°C" if units == "metric" else "°F"
        output = f"Weather Forecast for {city_name}, {country}:\n\n"
        
        for date, forecasts in list(daily_forecasts.items())[:5]:
            temps = [f["main"]["temp"] for f in forecasts]
            conditions = forecasts[len(forecasts)//2]["weather"][0]["description"]
            rain_prob = max(f.get("pop", 0) for f in forecasts) * 100
            
            output += f"📅 {date}:\n"
            output += f"   🌡️  {min(temps):.1f}-{max(temps):.1f}{unit_symbol}\n"
            output += f"   ☁️  {conditions.title()}\n"
            output += f"   🌧️  {rain_prob:.0f}% chance of rain\n\n"
        
        return output
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: City '{city}' not found. Please check the spelling."
        return f"Error: API request failed - {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: Network request failed - {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Unexpected API response format - {e}"
```

#### `src/agent/travel_agent.py`

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools.weather import get_weather_forecast

SYSTEM_PROMPT = """
You are a helpful travel planning assistant. When a user tells you about 
their upcoming trip, you should:

1. Use the weather forecast tool to get weather data for their destination
2. Analyze the forecast carefully
3. Provide:
   - A summary of expected weather conditions
   - A detailed packing list based on the weather
   - Activity recommendations suitable for the conditions

Packing guidelines:
- Below 10°C/50°F: Heavy coat, warm layers, gloves, scarf
- 10-20°C/50-68°F: Light jacket, layers, long pants
- 20-30°C/68-86°F: Light clothing, shorts, t-shirts
- Above 30°C/86°F: Very light clothing, sun protection critical
- Rain >30% probability: Umbrella, rain jacket, waterproof shoes
- Sunny conditions: Sunscreen, sunglasses, hat

Activity guidelines:
- Rainy days: Museums, galleries, cafes, shopping, cooking classes
- Sunny warm days: Parks, beaches, hiking, outdoor tours, markets
- Cold days: Hot springs, cozy restaurants, indoor attractions
- Extreme heat: Water activities, air-conditioned venues, early morning activities

Always be friendly, helpful, and provide practical, actionable advice!
"""


def create_travel_agent():
    """Create and return the travel planning agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    agent = create_react_agent(
        llm,
        tools=[get_weather_forecast],
        state_modifier=SYSTEM_PROMPT
    )
    
    return agent
```

#### `src/main.py`

```python
from dotenv import load_dotenv
load_dotenv()

from src.agent.travel_agent import create_travel_agent


def main():
    """Main entry point for the Smart Travel Planner."""
    print("🧳 Smart Travel Planner 🌤️")
    print("=" * 40)
    print("Tell me about your trip and I'll help you prepare!")
    print("(Type 'quit' to exit)\n")
    
    agent = create_travel_agent()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nHave a great trip! 👋✈️")
            break
        
        if not user_input:
            continue
        
        print("\nPlanning...\n")
        
        result = agent.invoke({
            "messages": [("user", user_input)]
        })
        
        # Extract the final message
        final_message = result["messages"][-1].content
        print(f"Travel Planner: {final_message}\n")


if __name__ == "__main__":
    main()
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# tests/test_weather_tool.py
import pytest
from unittest.mock import patch, MagicMock
from src.tools.weather import get_weather_forecast


class TestWeatherTool:
    """Test suite for weather tool."""
    
    @patch('src.tools.weather.requests.get')
    def test_successful_forecast(self, mock_get):
        """Test successful weather retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "city": {"name": "Paris", "country": "FR"},
            "list": [{
                "dt_txt": "2024-06-15 12:00:00",
                "main": {"temp": 22.5},
                "weather": [{"description": "clear sky"}],
                "pop": 0.1
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = get_weather_forecast.invoke({"city": "Paris"})
        
        assert "Paris" in result
        assert "FR" in result
    
    @patch('src.tools.weather.requests.get')
    def test_invalid_city(self, mock_get):
        """Test handling of invalid city name."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = \
            requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        result = get_weather_forecast.invoke({"city": "NotARealCity123"})
        
        assert "not found" in result.lower()
```

### 10.2 Integration Tests

```python
# tests/test_agent.py
import pytest
from src.agent.travel_agent import create_travel_agent


@pytest.mark.integration
class TestTravelAgent:
    """Integration tests for travel agent."""
    
    def test_agent_processes_trip_request(self):
        """Test that agent handles a basic trip request."""
        agent = create_travel_agent()
        
        result = agent.invoke({
            "messages": [("user", "I'm traveling to London next week")]
        })
        
        final_message = result["messages"][-1].content.lower()
        
        # Should mention weather-related advice
        assert any(word in final_message for word in 
                   ['weather', 'pack', 'recommend', 'suggest'])
```

---

## 11. 2-Hour Implementation Timeline

### Hour 1: Setup & Basic Tool (60 min)

| Time | Task | Deliverable |
|------|------|-------------|
| 0:00-0:15 | Environment Setup | `.env` file with API keys, `requirements.txt` installed |
| 0:15-0:30 | Get OpenWeatherMap API Key | Free account created, API key obtained |
| 0:30-0:50 | Implement Weather Tool | `weather.py` with `get_weather_forecast` function |
| 0:50-1:00 | Test Tool in Isolation | Verify tool returns correct data for test city |

### Hour 2: Agent Integration (60 min)

| Time | Task | Deliverable |
|------|------|-------------|
| 1:00-1:20 | Create ReAct Agent | `travel_agent.py` with system prompt |
| 1:20-1:35 | Integrate Tool with Agent | Agent successfully uses weather tool |
| 1:35-1:50 | Test with Various Prompts | Test different cities, edge cases |
| 1:50-2:00 | Polish & Demo Prep | Clean code, add comments, prepare demo |

---

## 12. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API Key Issues | Medium | High | Provide clear `.env.example`, test early |
| Rate Limiting | Low | Medium | Free tier is generous; add retry logic |
| Invalid City Names | Medium | Low | Graceful error handling with helpful messages |
| Network Failures | Low | Medium | Timeout handling, retry with backoff |
| JSON Parsing Errors | Low | Medium | Try-except with informative error messages |
| LLM API Costs | Medium | Low | Use gpt-4o-mini (cheap), limit testing |

---

## 13. Future Enhancements (v2.0+)

### P1 - High Value Extensions
- [ ] **Multi-city itinerary**: Handle trips with multiple destinations
- [ ] **Historical weather**: Compare with typical conditions for that time of year
- [ ] **Persistent memory**: Remember user preferences across sessions

### P2 - Nice to Have
- [ ] **Flight/hotel integration**: Combine with travel booking APIs
- [ ] **Local events**: Fetch events happening during travel dates
- [ ] **Currency exchange**: Show local currency info
- [ ] **Travel advisories**: Check government travel warnings

### P3 - Stretch Goals
- [ ] **Visual itinerary**: Generate PDF/image travel plan
- [ ] **Slack/Discord bot**: Deploy as chatbot
- [ ] **Mobile app**: React Native wrapper

---

## 14. Appendix

### A. Getting OpenWeatherMap API Key

1. Go to [https://openweathermap.org/api](https://openweathermap.org/api)
2. Click "Sign Up" and create a free account
3. Verify your email
4. Go to "API Keys" in your account dashboard
5. Copy your default API key (or generate a new one)
6. Add to your `.env` file: `OPENWEATHER_API_KEY=your_key_here`

**Note**: New API keys take ~10 minutes to activate!

### B. Sample Prompts for Testing

```
"I'm planning a trip to Tokyo from March 15-22"
"Help me pack for a week in Barcelona next month"
"What should I bring for a beach vacation in Miami?"
"I'm visiting London in December, what do I need?"
"Plan my trip to Reykjavik, Iceland in February"
```

### C. Common Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "API key not configured" | Missing .env | Add OPENWEATHER_API_KEY to .env |
| "City not found" | Typo or invalid city | Check spelling, use major city names |
| "401 Unauthorized" | Invalid API key | Verify key, wait 10 min if new |
| "429 Too Many Requests" | Rate limited | Wait 1 minute, reduce requests |

---

## 15. Sign-Off

| Role | Name | Date | Signature |
|------|------|------|----------|
| Author | | | |
| Reviewer | | | |
| Instructor | | | |

---

**Happy Travels! 🧳✈️🌍**
