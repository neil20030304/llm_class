2-Hour Agent Projects
Option 1: Wikipedia + DuckDuckGo
Project: "Research Assistant - Compare Sources"

Agent researches a topic using Wikipedia AND web search using DuckDuckGo

Compares/contrasts information from both sources

Generates brief report

Why this works:

Zero setup friction (no API keys for DDG)

Two tools reinforce same concept (information retrieval)

Shows value of multiple information sources

Students learn tool combination patterns

Learning outcomes:

Using multiple tools in sequence

Information synthesis from different sources

Basic agent reasoning about which tool to use when

Option 2: OpenWeatherMap
Project: "Smart Travel Planner"

Agent takes destination city and travel dates

Fetches weather forecast

Generates packing list and activity recommendations based on conditions

Why this works:

API key practice (important real-world skill)

Structured data handling (JSON parsing)

Clear cause-effect (weather → recommendations)

Real-world utility (students actually might use this!)

Learning outcomes:

API authentication patterns

Working with structured API responses

Conditional reasoning based on external data

Option 3: YouTube Transcript
Project: "Educational Video Analyzer"

Student provides educational video URL

Agent extracts transcript

Generates: summary, key concepts, quiz questions

See below for more details and starter code!

Why this works:

Reinforces custom tool creation (builds on calculator exercise)

High student engagement (analyze videos they actually watch)

Multiple LLM calls in sequence (fetch → summarize → generate questions)

Visible value (helps with studying!)

Learning outcomes:

Custom tool definition patterns

Multi-step agent workflows

Content transformation pipelines

Implementation Template (Works for Any of These)
Here's a 2-hour timeline that works for all three options:

Hour 1: Setup & Basic Tool

0-15 min: Install packages, get API keys (if needed)

15-45 min: Define custom tool OR use pre-built tool

45-60 min: Test tool in isolation (not in agent yet)

Hour 2: Agent Integration

60-90 min: Integrate tool with simple ReAct agent

90-110 min: Test with various prompts, debug

110-120 min: Optional extension or demo preparation

Detailed Recommendation: Start with YouTube
If I had to pick one for your next project after calculator, I'd choose YouTube Transcript.

Reasoning:

Builds directly on calculator pattern - they just wrote a custom tool, now they write another

No API key friction - can start coding immediately

High engagement - students love analyzing YouTube content

Clear progression - shows how custom tools enable new capabilities

Teaches real skill - most LangChain tools will be custom in production

Starter code for students:

python

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

@tool
def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video by video ID."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])

# Create agent with tool
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [get_youtube_transcript])

# Test it
result = agent.invoke({
    "messages": [("user", "Get the transcript for video dQw4w9WgXcQ and summarize it")]
})
Extensions for faster students:

Add summary formatting (bullet points, key quotes)

Extract chapter timestamps

Answer specific questions about video content

Compare transcripts from multiple videos

This gives you a solid 2-hour project that reinforces custom tool creation while introducing practical information retrieval patterns.

 

Apollo
