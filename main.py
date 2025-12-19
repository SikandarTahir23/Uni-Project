# main.py
import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")

# OpenAI client
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
model = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client)
config = RunConfig(model=model, tracing_disabled=True)

# Normal Python function for transcript
def get_youtube_transcript(url: str) -> str:
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    if not match:
        return "ERROR: Invalid YouTube URL"
    video_id = match.group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except (NoTranscriptFound, TranscriptsDisabled):
        return "ERROR: Transcript not available"
    return " ".join(item["text"] for item in transcript)

# AI Agent (without Chainlit-specific tool)
agent = Agent(
    name="YouTube Transcript AI Agent",
    instructions="You are a helpful assistant. Summarize or answer questions based on YouTube transcript.",
)
