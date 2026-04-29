# 2-Hour Project — Smart Travel Planner 🧳✈️

An AI-powered travel planning assistant built as the Topic 4 hands-on exercise. The agent takes a destination (and optional dates), pulls a real weather forecast from **OpenWeatherMap One Call API 3.0**, and produces a personalized **packing list + activity recommendations** based on the predicted conditions.

It's a small, end-to-end example of the patterns from Topic 4: a LangChain `@tool`, a `create_react_agent` core, and a thin LangGraph wrapper that owns the input/output loop.

---

## Why this project

The lecture compares two ways to wire tools into a LangGraph agent (manual `ToolNode` vs. `create_react_agent`). This project picks the **ReAct path** and uses it for something realistic, so the focus shifts from graph plumbing to the things you actually run into when shipping a tool-using agent:

- API authentication and `.env` handling
- Two-step API flow (geocode city → fetch forecast)
- Translating raw JSON into a prompt the LLM can reason over
- Conditional advice (temperature bands, rain probability)
- Conversational UX (verbose/quiet/help/exit commands)

---

## Architecture

```
        ┌─────────┐    user text     ┌──────────────┐
        │  input  │ ───────────────► │  call_agent  │
        │  node   │                  │ (ReAct loop) │
        └────▲────┘                  └──────┬───────┘
             │                              │
             │  loop back                   ▼
        ┌────┴────┐  prints reply   ┌──────────────┐
        │ output  │ ◄────────────── │ get_weather_ │
        │  node   │                 │   forecast   │
        └─────────┘                 └──────────────┘
```

- **`smart_travel_planner.py`** — the whole app:
  - `get_weather_forecast` — `@tool` that geocodes the city, calls OneCall 3.0, formats a 7-day forecast.
  - `create_travel_planner_agent()` — `create_react_agent(gpt-4o-mini, [weather_tool], SYSTEM_PROMPT)`.
  - `create_conversation_graph()` — wraps the agent in a `StateGraph` with `input → call_agent → output → input` and exit/verbose handling.
- **`PRD-SmartTravelPlanner.md`** — full product spec (problem, user stories, success criteria).
- **`SETUP_GUIDE.md`** — step-by-step setup for new students.
- **`test_weather_api.py`** — standalone smoke test for the weather tool (no LLM cost).
- **`requirements_travel_planner.txt`** — pinned deps.

---

## Quick start

```bash
cd topic4/2-hour-project
pip install -r requirements_travel_planner.txt

cp env_template.txt .env       # then fill in the two keys
# OPENAI_API_KEY=sk-...
# OPENWEATHER_API_KEY=...      # newly created keys take ~10 min to activate

python smart_travel_planner.py
```

In-session commands: `verbose`, `quiet`, `help`, `quit`.

---

## Examples

### Example 1 — Single-city trip

```
✈️  You: I'm planning a trip to Paris from June 15-20

🤖 Travel Planner:
🌤️  Paris is looking pleasant — daytime highs 22-26°C, mostly sunny, with a
    brief shower likely on the 17th (rain chance 55%).

📦 Pack:
  • Light layers (t-shirts + a cardigan for evenings ~15°C)
  • One pair of comfortable walking shoes
  • Compact umbrella for the 17th
  • SPF 30+, sunglasses

🎯 Suggested activities:
  • Outdoor: Luxembourg Gardens, Seine river walk, Montmartre on sunny days
  • Backup for the 17th: Louvre or Musée d'Orsay
  • Evening: open-air bistros — bring the cardigan
```

The agent calls `get_weather_forecast("Paris")` once, reads the 7-day forecast, then synthesizes the packing + activity advice in a single response.

### Example 2 — Extreme weather warning

```
✈️  You: Help me pack for Phoenix next week

🤖 Travel Planner:
⚠️  Heads up — Phoenix is hitting 41-44°C all week with UV index 11 (extreme).
    This is real heat, not "warm".

📦 Pack:
  • Loose, light, breathable fabrics only
  • Wide-brim hat + UV-blocking sunglasses
  • SPF 50+ (reapply every 2 hours outside)
  • Reusable water bottle — minimum 3L/day intake

🎯 Activities:
  • Outdoor only 6-9 AM or after sunset
  • Midday: Heard Museum, Musical Instrument Museum (both A/C)
  • Evening: Desert Botanical Garden after 6 PM
```

Demonstrates the system prompt's "above 30°C" + "UV high" branches firing together, plus the "extreme weather warning" guideline.

### Example 3 — Rainy / mixed forecast

```
✈️  You: What should I bring to London in late November?

🤖 Travel Planner:
🌧️  Cool and wet — 6-11°C, rain on 5 of 7 days (60-80% chance most afternoons).

📦 Pack:
  • Waterproof jacket (not just water-resistant)
  • Warm mid-layer (fleece or wool sweater)
  • Waterproof shoes + 2 extra pairs of wool socks
  • Compact travel umbrella

🎯 Activities:
  • British Museum, Tate Modern, V&A — all free, all indoors
  • Covered markets: Borough Market, Camden Stables
  • Pub lunches, afternoon tea
  • Save outdoor walks (Hyde Park, South Bank) for the drier mornings
```

Shows the agent reading the `pop` (rain probability) field across multiple days and shifting the recommendation toward indoor activities.

### Example 4 — Multi-turn follow-up

```
✈️  You: I'm going to Tokyo April 10-15
🤖 Travel Planner: [packing list + activities for cherry-blossom-season weather]

✈️  You: I forgot — I also want to do a day trip to Hakone. Same packing?
🤖 Travel Planner: Mostly yes, but Hakone is 600m higher, ~5°C cooler,
                   and rain is more likely. Add a warmer layer and waterproof shoes...
```

The conversation graph keeps message history across turns, so the agent reasons about the *delta* instead of starting over.

---

## What this project teaches

| Concept                  | Where it shows up                                              |
| ------------------------ | -------------------------------------------------------------- |
| `@tool` decorator        | `get_weather_forecast` — docstring becomes the tool schema     |
| API key hygiene          | `.env` + `python-dotenv`, `check_environment()` startup gate   |
| Two-step external call   | geocode → onecall, with separate error handling per step       |
| ReAct agent              | `create_react_agent(llm, [tool], prompt=SYSTEM_PROMPT)`        |
| Graph wrapper around ReAct | `input_node → call_agent_node → output_node` loop            |
| Special-input routing    | `__EXIT__`, `__VERBOSE__`, `__HELP__` sentinels in `route_after_input` |
| HTTP error mapping       | 401/404/429/timeout each return a user-friendly message        |

---

## Extension ideas

- Add a **currency converter** tool (multi-tool agent — exercises parallel dispatch from the lecture).
- Swap the manual ReAct wrapper for the explicit **`ToolNode`** version (Topic 4 Question 1) and compare graphs.
- Cache forecasts by `(city, date)` for the duration of a session.
- Wire in a flight/hotel search tool to make the planning end-to-end.

See `PRD-SmartTravelPlanner.md` for the full spec and `SETUP_GUIDE.md` for troubleshooting.
