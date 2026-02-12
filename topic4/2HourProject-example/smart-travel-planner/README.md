# 🧳 Smart Travel Planner 🌤️

An AI-powered travel planning assistant that uses weather forecasts to generate personalized packing lists and activity recommendations.

Built with **LangChain**, **LangGraph ReAct agents**, and the **OpenWeatherMap API**.

## ✨ Features

- 🌡️ **5-day weather forecasts** for any city worldwide
- 🎒 **Smart packing lists** based on temperature and conditions
- 🎯 **Activity recommendations** tailored to the weather
- 💬 **Interactive chat interface** for natural conversations
- 🔧 **Clean, modular code** following best practices

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd smart-travel-planner
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys
# - OPENWEATHER_API_KEY: Get from https://openweathermap.org/api
```

> ⚠️ **Note**: New OpenWeatherMap API keys take ~10 minutes to activate!

### 3. Run the Planner

```bash
# Interactive mode
python -m src.main

# Or with a single query
python -m src.main "Plan my trip to Paris next week"
```

## 📁 Project Structure

```
smart-travel-planner/
├── .env.example          # Template for environment variables
├── .gitignore
├── requirements.txt
├── pytest.ini            # Test configuration
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py           # Entry point & interactive CLI
│   ├── tools/
│   │   ├── __init__.py
│   │   └── weather.py    # OpenWeatherMap tool
│   ├── agent/
│   │   ├── __init__.py
│   │   └── travel_agent.py  # ReAct agent config
│   └── utils/
│       ├── __init__.py
│       └── config.py     # Environment management
└── tests/
    ├── __init__.py
    ├── test_weather_tool.py
    ├── test_config.py
    └── test_agent.py
```

## 💬 Example Usage

```
╔═══════════════════════════════════════════════════════════════╗
║          🧳 SMART TRAVEL PLANNER 🌤️                          ║
║       Your AI-powered trip preparation assistant              ║
╚═══════════════════════════════════════════════════════════════╝

You: I'm traveling to Tokyo from March 15-22

⏳ Planning your trip...

🤖 Travel Planner:

### 🌤️ Weather Summary
Tokyo will experience mild spring weather with temperatures ranging 
from 12-18°C (54-64°F). Expect partly cloudy skies with a 30% chance 
of rain on March 17-18.

### 🎒 Packing List
**Clothing:**
- Light jacket or fleece
- Long pants and jeans
- Layerable tops (t-shirts + sweaters)
- Comfortable walking shoes

**Accessories:**
- Compact umbrella
- Light scarf
- Sunglasses

**Essentials:**
- Sunscreen (SPF 30+)
- Portable phone charger
- Travel adapter

### 🎯 Activity Ideas
- Visit cherry blossom spots (peak season!)
- Explore Sensoji Temple and Asakusa
- Day trip to Mt. Fuji (clear days)
- Rainy day: TeamLab Borderless, Ghibli Museum

### ⚠️ Special Notes
March is cherry blossom season! Book accommodations early.
```

## 🧪 Running Tests

```bash
# Run all unit tests
pytest

# Run with coverage
pytest --cov=src

# Skip integration tests (require API keys)
pytest -m "not integration"

# Run only integration tests
pytest -m integration
```

## 🏗️ Architecture

```
┌─────────────────┐
│   User Input    │
│ "Plan my trip"  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ReAct Agent    │  ← LangGraph + GPT-4o-mini
│  (Reason + Act) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Weather Tool   │  ← Custom LangChain Tool
│  get_forecast() │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ OpenWeatherMap  │  ← External API
│     API         │
└─────────────────┘
```

## 🔑 Getting API Keys

### OpenAI API Key
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign in or create an account
3. Navigate to API Keys section
4. Create a new secret key
5. Copy to your `.env` file

### OpenWeatherMap API Key
1. Go to [openweathermap.org/api](https://openweathermap.org/api)
2. Sign up for a free account
3. Verify your email
4. Go to "API Keys" in your dashboard
5. Copy the default key (or create a new one)
6. Add to your `.env` file

> 💡 **Tip**: The free tier includes 60 calls/minute and 1M calls/month!

## 🎓 Learning Outcomes

This project demonstrates:

- **API Authentication**: Secure handling of API keys with environment variables
- **Tool Definition**: Creating custom LangChain tools with proper docstrings
- **ReAct Pattern**: Agents that Reason + Act in a loop
- **Error Handling**: Graceful handling of API errors and edge cases
- **Structured Data**: Parsing JSON responses from external APIs
- **Conditional Logic**: Making recommendations based on external data

## 🚧 Future Enhancements

- [ ] Multi-city itineraries
- [ ] Historical weather comparison
- [ ] Local events integration
- [ ] Export to PDF
- [ ] Slack/Discord bot

## 📝 License

MIT License - feel free to use this for learning!

---

Built with ❤️ for CS6501 - Exploring AI Agent Tools
