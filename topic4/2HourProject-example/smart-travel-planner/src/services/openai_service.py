"""OpenAI integration for generating AI-powered travel recommendations.

Keeps the OpenAI logic isolated from the Flask routes (SRP).
Returns structured data so the frontend can render it however it wants.
"""

import json
import os
import logging

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a concise, knowledgeable travel advisor. Given a city name and its
weather forecast data, return a JSON object with three keys:

1. "summary"  – A 2-3 sentence weather overview for the trip.
2. "packing"  – A flat list of 8-12 specific packing items suited to the
                forecast (strings only, no nested objects).
3. "activities" – A flat list of 6-10 weather-appropriate activity
                  suggestions for that destination (strings only).

Rules:
• Tailor every recommendation to the ACTUAL weather data provided.
• Include the destination's cultural flavour in activities (e.g. "Visit the
  Louvre" for Paris, not just "Visit a museum").
• If rain probability is high, bias towards indoor activities & rain gear.
• If temperatures are extreme, mention protective clothing first.
• Return ONLY valid JSON — no markdown fences, no commentary.
"""


def _get_client() -> OpenAI | None:
    """Create an OpenAI client if the API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        return None
    return OpenAI(api_key=api_key)


def generate_travel_recommendations(
    city: str,
    weather_text: str,
    units: str = "metric",
    start_date: str | None = None,
    end_date: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict | None:
    """
    Ask OpenAI to generate packing & activity recommendations.

    Args:
        city: Destination city name.
        weather_text: The formatted weather forecast string from our weather tool.
        units: 'metric' or 'imperial' — passed for context.
        start_date: Arrival date (YYYY-MM-DD) or None.
        end_date: Departure date (YYYY-MM-DD) or None.
        model: OpenAI model to use. gpt-4o-mini is cheap & fast.

    Returns:
        A dict with keys {"summary", "packing", "activities"} on success,
        or None if the call fails (caller should fall back to hardcoded logic).
    """
    client = _get_client()
    if client is None:
        logger.warning("OpenAI API key not configured — skipping AI recommendations.")
        return None

    unit_label = "Celsius" if units == "metric" else "Fahrenheit"

    date_line = ""
    if start_date and end_date:
        date_line = f"Travel dates: {start_date} to {end_date}\n"
    elif start_date:
        date_line = f"Arrival date: {start_date}\n"

    user_message = (
        f"Destination: {city}\n"
        f"Temperature units: {unit_label}\n"
        f"{date_line}\n"
        f"Weather forecast:\n{weather_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Validate expected keys exist
        if not all(k in data for k in ("summary", "packing", "activities")):
            logger.error("OpenAI response missing required keys: %s", data.keys())
            return None

        # Ensure lists are actually lists of strings
        data["packing"] = [str(item) for item in data["packing"]]
        data["activities"] = [str(item) for item in data["activities"]]
        data["summary"] = str(data["summary"])

        return data

    except (OpenAIError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.exception("OpenAI recommendation generation failed: %s", exc)
        return None
