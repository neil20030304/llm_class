#!/usr/bin/env python3
"""Main entry point for the Smart Travel Planner."""

import sys

from src.utils.config import load_config, validate_config
from src.agent.travel_agent import create_travel_agent, invoke_agent


BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║          🧳 SMART TRAVEL PLANNER 🌤️                          ║
║       Your AI-powered trip preparation assistant              ║
╚═══════════════════════════════════════════════════════════════╝
"""

EXAMPLES = """
📝 Example prompts:
   • "I'm traveling to Paris from June 15-20"
   • "Help me pack for a week in Tokyo"
   • "What should I bring for a beach trip to Miami?"
   • "Plan my winter trip to Reykjavik, Iceland"

💡 Tips:
   • Include your destination city and travel dates
   • Ask for specific advice (packing, activities, etc.)
   • Type 'quit' or 'exit' to leave
"""


def print_banner():
    """Display the application banner."""
    print(BANNER)


def check_configuration() -> bool:
    """
    Validate required API keys are configured.
    
    Returns:
        True if all keys are present, False otherwise.
    """
    config_status = validate_config()
    all_valid = all(config_status.values())
    
    if not all_valid:
        print("\n⚠️  Configuration Issue Detected:\n")
        for key, is_set in config_status.items():
            status = "✅" if is_set else "❌ MISSING"
            print(f"   {key}: {status}")
        print("\n📋 To fix this:")
        print("   1. Copy .env.example to .env")
        print("   2. Add your API keys to the .env file")
        print("   3. Restart the application\n")
        return False
    
    return True


def run_interactive_mode():
    """Run the travel planner in interactive mode."""
    print(EXAMPLES)
    print("=" * 65)
    print("\n🚀 Agent ready! Tell me about your trip:\n")
    
    # Create the agent
    agent = create_travel_agent()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
            break
        
        # Skip empty input
        if not user_input:
            continue
        
        # Process the request
        print("\n⏳ Planning your trip...\n")
        
        try:
            response = invoke_agent(agent, user_input)
            print(f"🤖 Travel Planner:\n\n{response}\n")
            print("-" * 65 + "\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("Please try again or check your API keys.\n")
    
    print("\n✈️  Have an amazing trip! Safe travels! 👋\n")


def run_single_query(query: str):
    """Run a single query and exit."""
    agent = create_travel_agent()
    response = invoke_agent(agent, query)
    print(response)


def main():
    """Main entry point."""
    # Load environment variables
    load_config()
    
    # Show banner
    print_banner()
    
    # Validate configuration
    if not check_configuration():
        sys.exit(1)
    
    # Check for command-line query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_single_query(query)
    else:
        # Interactive mode
        run_interactive_mode()


if __name__ == "__main__":
    main()
