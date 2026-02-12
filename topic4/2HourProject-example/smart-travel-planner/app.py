#!/usr/bin/env python3
"""Flask API server for Smart Travel Planner website.

Serves the static website AND provides API endpoints for weather data
and AI-powered travel recommendations.

Fallback chain for /api/travel-plan:
  1. LangGraph ReAct agent  (full AI reasoning + tool use)
  2. Weather tool + OpenAI   (structured JSON, no agent)
  3. Weather tool only        (frontend uses hardcoded logic)
"""

import logging
import os

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.tools.weather import get_weather_forecast
from src.services.openai_service import generate_travel_recommendations
from src.agent.travel_agent import (
    build_trip_message,
    get_or_create_agent,
    invoke_agent,
    is_agent_available,
)
from src.utils.response_parser import parse_agent_response

logger = logging.getLogger(__name__)

# Disable Flask's built-in static handler (static_folder=None) so that all
# file serving goes through our whitelisted serve_static route.
app = Flask(__name__, static_folder=None)
CORS(app)  # Enable CORS for local development


# ==========================================================================
# STATIC FILE SERVING
# ==========================================================================

@app.route('/')
def serve_index():
    """Serve the main index.html page."""
    return send_from_directory('.', 'index.html')


# Extensions that are safe to serve.  Everything else (e.g. .env, .py,
# .txt, .ini) gets a 404 so we never accidentally leak secrets.
ALLOWED_STATIC_EXT = {
    '.css', '.js', '.html', '.htm',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico',
    '.woff', '.woff2', '.ttf', '.eot',
    '.json', '.map',
}


@app.route('/<path:path>')
def serve_static(path):
    """Serve whitelisted static files only.

    Rejects dotfiles and any extension not in ALLOWED_STATIC_EXT so that
    sensitive files like .env, .py, or requirements.txt are never served.
    """
    # Block dotfiles (e.g. .env, .git/)
    if any(part.startswith('.') for part in path.split('/')):
        abort(404)

    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_STATIC_EXT:
        abort(404)

    return send_from_directory('.', path)


# ==========================================================================
# API ENDPOINTS
# ==========================================================================

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """
    Get weather forecast for a city.
    
    Query Parameters:
        city (str): Name of the city (required)
        units (str): 'metric' or 'imperial' (default: 'metric')
    
    Returns:
        JSON with weather data or error message
    """
    city = request.args.get('city', '').strip()
    units = request.args.get('units', 'metric')
    
    if not city:
        return jsonify({
            'error': 'City parameter is required',
            'example': '/api/weather?city=Paris&units=metric'
        }), 400
    
    if units not in ['metric', 'imperial']:
        units = 'metric'
    
    try:
        # Call our weather tool
        result = get_weather_forecast.invoke({
            'city': city,
            'units': units
        })
        
        # Check if result contains an error
        if result.startswith('❌'):
            return jsonify({
                'error': result.replace('❌ ', ''),
                'city': city
            }), 400
        
        return jsonify({
            'success': True,
            'city': city,
            'units': units,
            'weather': result
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to fetch weather: {str(e)}',
            'city': city
        }), 500


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validated_params():
    """Extract & validate common query parameters (city, units, dates)."""
    city = request.args.get('city', '').strip()
    units = request.args.get('units', 'metric')
    start_date = request.args.get('start_date', '').strip() or None
    end_date = request.args.get('end_date', '').strip() or None

    if units not in ('metric', 'imperial'):
        units = 'metric'

    return city, units, start_date, end_date


def _fetch_weather(city: str, units: str):
    """Call the weather tool and return (text, error_response)."""
    try:
        text = get_weather_forecast.invoke({'city': city, 'units': units})
    except Exception as e:
        return None, (jsonify({'error': f'Failed to fetch weather: {e}', 'city': city}), 500)

    if text.startswith('❌'):
        return None, (jsonify({'error': text.replace('❌ ', ''), 'city': city}), 400)

    return text, None


# ---------------------------------------------------------------------------
# /api/travel-plan  —  the main endpoint the frontend uses
# ---------------------------------------------------------------------------

@app.route('/api/travel-plan', methods=['GET'])
def get_travel_plan():
    """
    AI-powered travel plan: weather + packing + activities.

    Query Parameters:
        city (str): Destination city (required)
        units (str): 'metric' or 'imperial' (default: 'metric')
        start_date (str): Arrival date, YYYY-MM-DD (optional)
        end_date (str): Departure date, YYYY-MM-DD (optional)

    Fallback chain:
        1. LangGraph ReAct agent   → structured via response_parser
        2. Weather + OpenAI service → structured JSON
        3. Weather only             → frontend hardcoded logic
    """
    city, units, start_date, end_date = _validated_params()

    if not city:
        return jsonify({
            'error': 'City parameter is required',
            'example': '/api/travel-plan?city=Paris&units=metric'
        }), 400

    base = {
        'city': city,
        'units': units,
        'start_date': start_date,
        'end_date': end_date,
    }

    # --- Path 1: LangGraph Agent -------------------------------------------
    if is_agent_available():
        try:
            agent = get_or_create_agent()
            message = build_trip_message(city, units, start_date, end_date)
            raw = invoke_agent(agent, message)
            parsed = parse_agent_response(raw)

            # Fetch the formatted day-by-day forecast for the weather panel.
            # The agent consumed weather data internally, but its final
            # response is prose — we still want the detailed table.
            weather_text, _ = _fetch_weather(city, units)

            return jsonify({
                **base,
                'success': True,
                'ai_powered': True,
                'summary': parsed['summary'],
                'packing': parsed['packing'],
                'activities': parsed['activities'],
                'weather': weather_text or '',
            })
        except Exception:
            logger.exception('Agent path failed — falling back.')

    # --- Path 2: Weather + OpenAI service ----------------------------------
    weather_text, err = _fetch_weather(city, units)
    if err:
        return err

    ai_recs = generate_travel_recommendations(
        city, weather_text, units, start_date, end_date,
    )

    if ai_recs:
        return jsonify({
            **base,
            'success': True,
            'ai_powered': True,
            'weather': weather_text,
            'summary': ai_recs['summary'],
            'packing': ai_recs['packing'],
            'activities': ai_recs['activities'],
        })

    # --- Path 3: Weather only (frontend hardcoded fallback) ----------------
    return jsonify({
        **base,
        'success': True,
        'ai_powered': False,
        'weather': weather_text,
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    weather_key = bool(os.getenv('OPENWEATHER_API_KEY'))
    openai_key = bool(os.getenv('OPENAI_API_KEY'))
    return jsonify({
        'status': 'healthy',
        'weather_api_configured': weather_key,
        'openai_api_configured': openai_key,
    })


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == '__main__':
    # Check for API key
    if not os.getenv('OPENWEATHER_API_KEY'):
        print("\n⚠️  Warning: OPENWEATHER_API_KEY not set!")
        print("   Copy .env.example to .env and add your API key.")
        print("   Get a free key at: https://openweathermap.org/api\n")

    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Warning: OPENAI_API_KEY not set!")
        print("   AI recommendations will be disabled (hardcoded fallback used).")
        print("   Get a key at: https://platform.openai.com/api-keys\n")
    
    print("\n🌤️  TripCast - Smart Travel Planner")
    print("=" * 40)
    print("\n🚀 Server starting...")
    print("   Website: http://localhost:3000")
    print("   API:     http://localhost:3000/api/weather?city=Paris")
    print("\n   Press Ctrl+C to stop.\n")
    
    app.run(
        host='0.0.0.0',
        port=3000,
        debug=True
    )
