# Smart Travel Planner - Setup Guide

Quick setup guide for the Smart Travel Planner project.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Internet connection

## Step 1: Get API Keys

### OpenAI API Key

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)
5. **Important**: Save it immediately - you can't view it again!

### OpenWeatherMap API Key

1. Go to [https://openweathermap.org/api](https://openweathermap.org/api)
2. Click "Sign Up" (free tier is sufficient)
3. Verify your email
4. Go to "API Keys" in your account dashboard
5. Copy your default API key
6. **Note**: New keys take ~10 minutes to activate!

## Step 2: Install Dependencies

```bash
# Navigate to the project directory
cd topic4/

# Install required packages
pip install -r requirements_travel_planner.txt
```

## Step 3: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Use your favorite text editor (nano, vim, vscode, etc.)
nano .env
```

Your `.env` file should look like:

```bash
OPENAI_API_KEY=sk-proj-abc123your-actual-key-here
OPENWEATHER_API_KEY=your-actual-openweathermap-key-here
```

**Security**: Never commit `.env` to git! It's already in `.gitignore`.

## Step 4: Run the Application

```bash
python smart_travel_planner.py
```

## Usage Examples

Once the application starts, try these prompts:

```
I'm planning a trip to Paris from June 15-20

Help me pack for Tokyo next month

What should I bring for a beach vacation in Miami?

I'm visiting London in December, what do I need?
```

## Commands

- **quit/exit/q** - Exit the program
- **verbose** - Enable debug output
- **quiet** - Disable debug output  
- **help/?** - Show help message

## Troubleshooting

### "API key not configured"
- Check that your `.env` file exists in the same directory
- Verify the keys are correct (no extra spaces)
- Make sure the file is named exactly `.env` (not `.env.txt`)

### "City not found"
- Check spelling of city name
- Try adding country code: `"Paris,FR"` or `"London,UK"`
- Use major city names (smaller towns may not be in the database)

### "401 Unauthorized" (OpenWeatherMap)
- Your API key may be invalid
- If newly created, wait 10-15 minutes for activation
- Verify you copied the entire key

### "Rate limit exceeded"
- Free tier: 60 calls/minute for OpenWeatherMap
- Wait a minute and try again
- Consider upgrading if you need more calls

### Network errors
- Check your internet connection
- Try with verbose mode: type `verbose` after starting
- Some firewalls may block API calls

## Testing Without API Keys

If you want to test the code structure without API keys:

```python
# Comment out the check_environment() call in main()
# The tool will return error messages but the code will run
```

## Next Steps

- Read the full PRD: `PRD-SmartTravelPlanner.md`
- Compare with examples: `react_agent_example.py`, `tool_example.py`
- Extend with additional tools (e.g., currency converter, flight search)

## Cost Estimates

- **OpenAI GPT-4o-mini**: ~$0.0001 per request (very cheap!)
- **OpenWeatherMap Free Tier**: Unlimited calls (within rate limits)
- **Expected cost per session**: < $0.01

Happy travels! 🧳✈️🌍



