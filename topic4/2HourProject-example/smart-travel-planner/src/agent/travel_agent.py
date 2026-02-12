"""Travel planning agent using ReAct pattern with LangGraph.

Provides both a CLI-friendly interface (invoke_agent) and a
web-friendly interface (get_or_create_agent / build_trip_message)
so the Flask app can use the same agent that powers the CLI.
"""

import logging
import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.tools.weather import get_weather_forecast

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a friendly and helpful travel planning assistant! 🧳✈️

When a user tells you about their upcoming trip, follow these steps:

1. **Get the weather**: Use the get_weather_forecast tool to fetch weather data 
   for their destination city. If the user specifies temperature units
   (Celsius or Fahrenheit), pass the corresponding `units` parameter to the
   tool: "metric" for Celsius, "imperial" for Fahrenheit.

2. **Analyze conditions**: Look at temperature ranges, precipitation chances, 
   and general conditions across their travel dates.  If the user provides
   specific travel dates, focus your advice on those dates.  If the dates
   fall outside the 5-day forecast window, note that and give general
   seasonal guidance.

3. **Provide recommendations**: Give them:
   - A brief weather summary
   - A detailed packing list based on conditions
   - Activity suggestions appropriate for the weather *and* destination

## Packing Guidelines by Temperature:

🥶 **Cold (Below 10°C / 50°F)**:
- Heavy winter coat, thermal layers, warm sweater
- Gloves, scarf, warm hat/beanie
- Warm boots, thick socks
- Hand warmers (optional)

🍂 **Cool (10-20°C / 50-68°F)**:
- Light jacket or fleece
- Long pants, jeans
- Layers (t-shirt + sweater)
- Closed-toe shoes

☀️ **Warm (20-30°C / 68-86°F)**:
- Light clothing, shorts, t-shirts
- Sundress or light pants
- Sandals and comfortable walking shoes
- Light cardigan for air conditioning

🔥 **Hot (Above 30°C / 86°F)**:
- Very light, breathable fabrics
- Shorts, tank tops, linen
- Wide-brim hat for sun protection
- Light-colored clothing

## Weather-Specific Items:

🌧️ **Rain (>30% chance)**:
- Umbrella (compact travel size)
- Rain jacket or waterproof layer
- Waterproof shoes or shoe covers
- Quick-dry clothing

☀️ **Sunny conditions**:
- Sunscreen (SPF 30+)
- Sunglasses
- Hat or cap
- Lip balm with SPF

💨 **Windy conditions (>20 km/h)**:
- Windbreaker
- Hair ties / headband
- Secure hat

## Activity Recommendations:

🌧️ **Rainy / Bad Weather**:
- Museums and galleries
- Local cafes and restaurants
- Shopping districts
- Cooking classes
- Spa and wellness
- Indoor markets
- Movie theaters
- Escape rooms

☀️ **Sunny / Good Weather**:
- Parks and gardens
- Walking tours
- Beaches (if applicable)
- Hiking and nature trails
- Outdoor markets
- Rooftop bars
- Boat tours
- Picnics

❄️ **Cold Weather**:
- Hot springs / thermal baths
- Cozy cafes
- Winter sports (if available)
- Indoor attractions
- Warm restaurant hopping

🔥 **Extreme Heat**:
- Water activities (pools, beaches)
- Early morning sightseeing
- Air-conditioned museums
- Evening outdoor activities
- Indoor malls

## Response Format:

Always structure your response like this:

### 🌤️ Weather Summary
[Brief overview of expected conditions]

### 🎒 Packing List
**Clothing:**
- [items]

**Accessories:**
- [items]

**Essentials:**
- [items]

### 🎯 Activity Ideas
[Weather-appropriate AND destination-specific suggestions — mention real
landmarks, neighbourhoods, restaurants, etc. for the given city]

### ⚠️ Special Notes
[Any warnings about extreme weather, events, or tips]

---

Be enthusiastic, helpful, and practical! Use emojis to make your responses 
friendly and scannable. If you're unsure about the city name, ask for clarification.
"""


def create_travel_agent(model: str = "gpt-4o-mini", temperature: float = 0.7):
    """
    Create and configure the travel planning agent.

    Args:
        model: OpenAI model to use. Defaults to gpt-4o-mini for cost efficiency.
        temperature: Creativity setting (0-1). Higher = more creative responses.

    Returns:
        Configured ReAct agent ready to handle travel planning queries.
    """
    llm = ChatOpenAI(model=model, temperature=temperature)

    agent = create_react_agent(
        llm,
        tools=[get_weather_forecast],
        prompt=SYSTEM_PROMPT,
    )

    return agent


def invoke_agent(agent, user_message: str) -> str:
    """
    Send a message to the agent and get the response.

    Args:
        agent: The travel planning agent instance.
        user_message: User's travel query.

    Returns:
        The agent's response as a string.
    """
    result = agent.invoke({
        "messages": [("user", user_message)]
    })

    return result["messages"][-1].content


# =========================================================================
# Web-server helpers
# =========================================================================

_cached_agent = None


def get_or_create_agent():
    """Return a lazily-created, module-level cached agent instance.

    Creating the agent is cheap (no API call), but we still avoid doing it
    on every request.
    """
    global _cached_agent  # noqa: PLW0603
    if _cached_agent is None:
        logger.info("Creating LangGraph travel agent…")
        _cached_agent = create_travel_agent()
    return _cached_agent


def build_trip_message(
    city: str,
    units: str = "metric",
    start_date: str | None = None,
    end_date: str | None = None,
) -> str:
    """Build a natural-language query from form fields.

    The resulting message is what the agent receives as if a human typed it.
    """
    unit_label = "Celsius" if units == "metric" else "Fahrenheit"
    parts = [f"I'm planning a trip to {city}."]

    if start_date and end_date:
        parts.append(f"I'll be there from {start_date} to {end_date}.")
    elif start_date:
        parts.append(f"I'm arriving on {start_date}.")
    elif end_date:
        parts.append(f"I need to be there by {end_date}.")

    parts.append(f"Please use {unit_label} for temperatures.")
    parts.append(
        "What's the weather forecast? What should I pack, "
        "and what activities do you recommend?"
    )
    return " ".join(parts)


def is_agent_available() -> bool:
    """Return True when the OPENAI_API_KEY looks usable."""
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith("sk-your")
