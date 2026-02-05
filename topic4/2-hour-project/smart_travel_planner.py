"""
Smart Travel Planner Agent
==========================

An AI-powered travel planning assistant that uses OpenWeatherMap API to fetch
real-time weather forecasts and provides personalized packing lists and activity
recommendations for your trip.

Requirements:
    - pip install langchain langchain-openai langgraph requests python-dotenv

Environment Variables (.env file):
    OPENAI_API_KEY=your-openai-key-here
    OPENWEATHER_API_KEY=your-openweathermap-key-here

Usage:
    python smart_travel_planner.py

Author: CS6501 - Topic 4: Exploring Tools
Version: 1.0
"""

import os
import requests
from typing import Annotated, Sequence, Literal
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from typing import TypedDict

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
OPENWEATHER_ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

# Agent system prompt with packing and activity guidelines
SYSTEM_PROMPT = """
You are a helpful and enthusiastic travel planning assistant! 🧳✈️

When a user tells you about their upcoming trip, you should:

1. **Use the weather forecast tool** to get weather data for their destination
2. **Analyze the forecast carefully** - look at temperatures, conditions, and rain probability
3. **Provide comprehensive advice** including:
   - A friendly summary of expected weather conditions
   - A detailed, practical packing list based on the weather
   - Activity recommendations suitable for the conditions
   - Any warnings about extreme weather

**Packing Guidelines:**
- Below 10°C/50°F: Heavy coat, warm layers, thermal underwear, gloves, scarf, winter boots
- 10-20°C/50-68°F: Light jacket or cardigan, layers, long pants, closed-toe shoes
- 20-30°C/68-86°F: Light clothing, t-shirts, shorts, comfortable walking shoes, light fabrics
- Above 30°C/86°F: Very light, breathable clothing, sun protection CRITICAL, hydration supplies
- Rain >30% probability: Compact umbrella, rain jacket/poncho, waterproof shoes or shoe covers
- Rain >60% probability: Full rain gear, waterproof bag covers, extra pairs of socks
- Sunny conditions: SPF 50+ sunscreen, UV-blocking sunglasses, wide-brimmed hat, light long sleeves
- High humidity: Breathable, moisture-wicking fabrics, extra changes of clothes
- Windy conditions: Windbreaker, secure hat with chin strap

**Activity Guidelines:**
- **Rainy days**: 
  * Indoor museums and art galleries
  * Cozy cafes and local cuisine tasting
  * Shopping malls and boutiques
  * Cooking classes or workshops
  * Spa and wellness centers
  * Theater shows or cinema

- **Sunny warm days (20-30°C/68-86°F)**: 
  * Parks and botanical gardens
  * Beach activities (if coastal)
  * Hiking and nature trails
  * Outdoor markets and street food tours
  * Bike tours around the city
  * Open-air concerts or festivals

- **Cold days (below 10°C/50°F)**: 
  * Hot springs or thermal baths
  * Cozy restaurants with local specialties
  * Indoor historical sites
  * Christmas markets (seasonal)
  * Museums with extensive indoor exhibits
  * Coffee shop hopping

- **Extreme heat (above 30°C/86°F)**: 
  * Water activities (pools, water parks, beaches)
  * Air-conditioned shopping centers
  * Early morning activities (6-9 AM)
  * Evening strolls after sunset
  * Indoor attractions
  * Take frequent breaks in shade

**Communication Style:**
- Be friendly, enthusiastic, and encouraging
- Use emojis sparingly for visual appeal
- Provide specific, actionable advice
- Organize information clearly with sections
- Mention any extreme weather warnings prominently
- If the forecast shows changing conditions, advise packing for variety

Always be practical and help travelers feel prepared and excited for their trip!
"""

# ============================================================================
# WEATHER TOOL
# ============================================================================

@tool
def get_weather_forecast(city: str, units: str = "metric") -> str:
    """
    Fetch 8-day weather forecast for a specified city using OpenWeatherMap One Call API 3.0.
    
    This tool first geocodes the city name to get coordinates, then retrieves detailed 
    weather information including current conditions and daily forecasts.
    
    Args:
        city: Name of the city (e.g., "Paris", "Tokyo", "New York", "Los Angeles")
              Can include country code for disambiguation (e.g., "Paris,FR")
        units: Temperature units - "metric" (Celsius) or "imperial" (Fahrenheit)
               Default is "metric"
    
    Returns:
        Formatted string containing:
        - City name and country
        - Current weather conditions
        - 7-day forecast with dates
        - Temperature ranges (min-max)
        - Weather conditions description
        - Precipitation probability
        
        Or an error message if the request fails.
    
    Examples:
        >>> get_weather_forecast("London")
        >>> get_weather_forecast("New York", "imperial")
        >>> get_weather_forecast("Tokyo,JP")
    """
    from datetime import datetime
    
    # Validate API key
    if not OPENWEATHER_API_KEY:
        return (
            "❌ Error: OpenWeatherMap API key not configured.\n"
            "Please set OPENWEATHER_API_KEY in your .env file.\n"
            "Get a free API key at: https://openweathermap.org/api"
        )
    
    try:
        # Step 1: Geocode the city to get lat/lon coordinates
        geo_response = requests.get(
            OPENWEATHER_GEO_URL,
            params={
                "q": city,
                "limit": 1,
                "appid": OPENWEATHER_API_KEY
            },
            timeout=10
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            return (
                f"❌ Error: City '{city}' not found.\n"
                f"Please check the spelling or try adding a country code (e.g., 'Paris,FR').\n"
                f"Make sure to use major city names."
            )
        
        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        city_name = geo_data[0]["name"]
        country = geo_data[0].get("country", "")
        
        # Step 2: Fetch weather data using One Call API 3.0
        weather_response = requests.get(
            OPENWEATHER_ONECALL_URL,
            params={
                "lat": lat,
                "lon": lon,
                "exclude": "minutely,alerts",
                "units": units,
                "appid": OPENWEATHER_API_KEY
            },
            timeout=10
        )
        weather_response.raise_for_status()
        data = weather_response.json()
        
        # Format output
        unit_symbol = "°C" if units == "metric" else "°F"
        speed_unit = "m/s" if units == "metric" else "mph"
        
        output_lines = [
            f"🌍 Weather Forecast for {city_name}, {country}",
            "=" * 50,
            ""
        ]
        
        # Current weather
        if data.get("current"):
            current = data["current"]
            output_lines.extend([
                "🌤️  CURRENT CONDITIONS:",
                f"   🌡️  Temperature: {current['temp']:.1f}{unit_symbol}",
                f"   🤔 Feels Like: {current['feels_like']:.1f}{unit_symbol}",
                f"   ☁️  Conditions: {current['weather'][0]['description'].title()}",
                f"   💧 Humidity: {current['humidity']}%",
                f"   💨 Wind: {current['wind_speed']:.1f} {speed_unit}",
                f"   ☀️  UV Index: {current.get('uvi', 'N/A')}",
                ""
            ])
        
        output_lines.append("📅 DAILY FORECAST:")
        output_lines.append("")
        
        # Daily forecast (up to 7 days)
        if data.get("daily"):
            for day in data["daily"][:7]:
                date = datetime.fromtimestamp(day["dt"]).strftime("%Y-%m-%d (%a)")
                temp_min = day["temp"]["min"]
                temp_max = day["temp"]["max"]
                feels_min = day["feels_like"].get("day", temp_min)
                conditions = day["weather"][0]["description"].title()
                weather_main = day["weather"][0]["main"]
                rain_prob = day.get("pop", 0) * 100
                humidity = day["humidity"]
                wind_speed = day["wind_speed"]
                uvi = day.get("uvi", "N/A")
                
                output_lines.extend([
                    f"📅 {date}:",
                    f"   🌡️  Temperature: {temp_min:.1f} - {temp_max:.1f}{unit_symbol}",
                    f"   ☁️  Conditions: {conditions} ({weather_main})",
                    f"   🌧️  Rain Chance: {rain_prob:.0f}%",
                    f"   💧 Humidity: {humidity}%",
                    f"   💨 Wind: {wind_speed:.1f} {speed_unit}",
                    f"   ☀️  UV Index: {uvi}",
                    ""
                ])
        
        output_lines.append("=" * 50)
        return "\n".join(output_lines)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return (
                f"❌ Error: City '{city}' not found.\n"
                f"Please check the spelling or try adding a country code (e.g., 'Paris,FR').\n"
                f"Make sure to use major city names."
            )
        elif e.response.status_code == 401:
            return (
                f"❌ Error: Invalid API key or subscription required.\n"
                f"The One Call API 3.0 may require a subscription.\n"
                f"Please verify your OPENWEATHER_API_KEY and check your subscription at:\n"
                f"https://openweathermap.org/api/one-call-3\n"
                f"Note: New API keys take ~10 minutes to activate after creation."
            )
        elif e.response.status_code == 429:
            return (
                f"❌ Error: API rate limit exceeded.\n"
                f"Please wait a minute before making another request."
            )
        else:
            return f"❌ Error: API request failed - HTTP {e.response.status_code}: {e}"
            
    except requests.exceptions.Timeout:
        return (
            "❌ Error: Request timed out.\n"
            "Please check your internet connection and try again."
        )
        
    except requests.exceptions.RequestException as e:
        return f"❌ Error: Network request failed - {e}"
        
    except (KeyError, IndexError, ValueError) as e:
        return (
            f"❌ Error: Unexpected API response format.\n"
            f"Details: {e}\n"
            f"The OpenWeatherMap API may have changed. Please report this issue."
        )


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ConversationState(TypedDict):
    """
    State schema for the travel planner conversation.
    
    Attributes:
        messages: Full conversation history with automatic message merging
        verbose: Controls detailed tracing output (for debugging)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    verbose: bool


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_travel_planner_agent():
    """
    Create and return the travel planning ReAct agent.
    
    The agent uses:
    - GPT-4o-mini for cost-effective reasoning
    - Weather forecast tool for external data
    - ReAct pattern (Reasoning + Acting) for tool usage
    
    Returns:
        Compiled LangGraph agent ready for invocation
    """
    # Check if OpenAI API key is configured
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found!\n"
            "Please set OPENAI_API_KEY in your .env file.\n"
            "Get an API key at: https://platform.openai.com/api-keys"
        )
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective model
        temperature=0.7,       # Balanced creativity and consistency
    )
    
    # Create the ReAct agent with weather tool
    agent = create_react_agent(
        model=llm,
        tools=[get_weather_forecast],
        prompt=SYSTEM_PROMPT  # Inject system prompt
    )
    
    return agent


# ============================================================================
# CONVERSATION WRAPPER GRAPH
# ============================================================================

def create_conversation_graph():
    """
    Build a conversation wrapper around the ReAct agent.
    
    This graph handles:
    - User input collection
    - Agent invocation
    - Output display
    - Conversation looping
    
    Graph structure:
        input → call_agent → output → input (loop)
                    ↓
                   END (on quit)
    
    Returns:
        Compiled conversation graph
    """
    
    # Create the ReAct agent
    travel_agent = create_travel_planner_agent()
    
    def input_node(state: ConversationState) -> ConversationState:
        """
        Get input from user.
        
        Handles:
        - Normal user queries
        - Quit commands (exit, quit, q)
        - Verbose toggle commands
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("\n[DEBUG] Entering input_node")
        
        # Prompt for input
        user_input = input("\n✈️  You: ").strip()
        
        # Handle quit commands
        if user_input.lower() in ["quit", "exit", "q", "bye"]:
            if verbose:
                print("[DEBUG] Exit command received")
            return {"messages": [HumanMessage(content="__EXIT__")]}
        
        # Handle verbose toggle
        if user_input.lower() == "verbose":
            print("🔊 Verbose mode enabled - Debug output will be shown")
            return {"verbose": True, "messages": [HumanMessage(content="__VERBOSE__")]}
        
        if user_input.lower() == "quiet":
            print("🔇 Quiet mode enabled - Debug output hidden")
            return {"verbose": False, "messages": [HumanMessage(content="__QUIET__")]}
        
        # Handle help command
        if user_input.lower() in ["help", "?"]:
            help_text = """
📖 Smart Travel Planner - Help

Available Commands:
  • Just tell me about your trip! For example:
    - "I'm planning a trip to Paris from June 15-20"
    - "Help me pack for Tokyo next month"
    - "What's the weather like in Barcelona?"
  
  • quit/exit/q - Exit the program
  • verbose - Enable debug output
  • quiet - Disable debug output
  • help/? - Show this help message

Tips:
  • Include the city name and dates for best results
  • Be specific about your travel dates
  • You can ask follow-up questions
            """
            print(help_text)
            return {"messages": [HumanMessage(content="__HELP__")]}
        
        # Add user message to history
        if verbose:
            print(f"[DEBUG] User input: {user_input}")
        
        return {"messages": [HumanMessage(content=user_input)]}
    
    
    def call_agent_node(state: ConversationState) -> ConversationState:
        """
        Invoke the travel planner agent.
        
        The agent will:
        1. Analyze the user's request
        2. Decide if it needs weather data
        3. Use the weather tool if needed
        4. Generate a response with recommendations
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("[DEBUG] Entering call_agent_node")
            print(f"[DEBUG] Message history length: {len(state['messages'])}")
        
        # Invoke the ReAct agent
        result = travel_agent.invoke({"messages": state["messages"]})
        
        if verbose:
            print(f"[DEBUG] Agent returned {len(result['messages'])} messages")
        
        # Return only the new messages
        return {"messages": result["messages"][len(state['messages']):]}
    
    
    def output_node(state: ConversationState) -> ConversationState:
        """
        Display the agent's response to the user.
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("[DEBUG] Entering output_node")
        
        # Find the last AI message
        last_ai_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                last_ai_message = msg
                break
        
        if last_ai_message:
            print(f"\n🤖 Travel Planner:\n{last_ai_message.content}\n")
        else:
            print("\n⚠️  (No response generated)\n")
        
        return {}
    
    
    def route_after_input(state: ConversationState) -> Literal["call_agent", "output", "end"]:
        """
        Route based on user input.
        
        - __EXIT__ → end
        - __VERBOSE__/__QUIET__/__HELP__ → output (skip agent)
        - Normal input → call_agent
        """
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'content'):
            content = last_message.content
            if content == "__EXIT__":
                return "end"
            elif content in ["__VERBOSE__", "__QUIET__", "__HELP__"]:
                return "output"  # Skip to output (don't call agent)
        
        return "call_agent"
    
    
    # Build the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("input", input_node)
    workflow.add_node("call_agent", call_agent_node)
    workflow.add_node("output", output_node)
    
    # Set entry point
    workflow.set_entry_point("input")
    
    # Add edges
    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {
            "call_agent": "call_agent",
            "output": "output",
            "end": END
        }
    )
    workflow.add_edge("call_agent", "output")
    workflow.add_edge("output", "input")  # Loop back
    
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🧳 Smart Travel Planner 🌤️                          ║
║                                                              ║
║      AI-Powered Weather-Based Travel Recommendations         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Welcome! I'm your AI travel planning assistant. Tell me about your
upcoming trip and I'll help you prepare with:

  📦 Personalized packing lists
  🎯 Activity recommendations  
  🌡️  Real-time weather forecasts
  ⚠️  Important weather warnings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 Example prompts:
  • "I'm traveling to Paris from June 15-20"
  • "Help me pack for a beach trip to Miami next week"
  • "What should I bring to Tokyo in March?"

💡 Commands: quit, help, verbose, quiet

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(banner)


def check_environment():
    """
    Check if required environment variables are set.
    
    Raises:
        SystemExit: If critical environment variables are missing
    """
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append(
            "❌ OPENAI_API_KEY not found\n"
            "   Get one at: https://platform.openai.com/api-keys"
        )
    
    if not OPENWEATHER_API_KEY:
        errors.append(
            "❌ OPENWEATHER_API_KEY not found\n"
            "   Get one at: https://openweathermap.org/api"
        )
    
    if errors:
        print("\n⚠️  Environment Configuration Error\n")
        print("Missing required API keys:\n")
        for error in errors:
            print(error)
        print("\nPlease create a .env file with the following format:")
        print("─" * 50)
        print("OPENAI_API_KEY=your-openai-key-here")
        print("OPENWEATHER_API_KEY=your-openweathermap-key-here")
        print("─" * 50)
        print("\nNote: OpenWeatherMap API keys take ~10 minutes to activate.")
        raise SystemExit(1)


def main():
    """
    Main entry point for the Smart Travel Planner.
    
    Creates the conversation graph and runs the main loop.
    The graph handles all interaction internally via edge-based looping.
    """
    try:
        # Check environment configuration
        check_environment()
        
        # Print welcome banner
        print_banner()
        
        # Create the conversation graph
        print("🔧 Initializing travel planner agent...")
        app = create_conversation_graph()
        print("✅ Ready! Let's plan your trip.\n")
        
        # Initial state
        initial_state = {
            "messages": [],
            "verbose": False
        }
        
        # Run the graph (loops internally until user quits)
        app.invoke(initial_state)
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you're having trouble, try:")
        print("  1. Check your .env file has valid API keys")
        print("  2. Verify your internet connection")
        print("  3. Run with 'verbose' mode for debugging")
    
    print("\n✈️  Thanks for using Smart Travel Planner!")
    print("Have a wonderful trip! 🌍🧳\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

