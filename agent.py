# agent.py

import os
import re
from dotenv import load_dotenv

import chainlit as cl
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

from agents import (
    Agent,
    function_tool,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
)

from openai.types.responses import ResponseTextDeltaEvent

# -------------------------------------------------
# Load Environment
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing")

# -------------------------------------------------
# OpenAI Client & Model
# -------------------------------------------------
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=client,
)

config = RunConfig(model=model, tracing_disabled=True)

# -------------------------------------------------
# Tool: Transcript Fetcher (ROBUST)
# -------------------------------------------------
@function_tool
def fetch_youtube_transcript(url: str) -> str:
    """
    Fetch transcript safely with language fallback
    """
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    if not match:
        return "ERROR: Invalid YouTube URL."

    video_id = match.group(1)

    try:
        # Try English first
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US"]
        )
    except (NoTranscriptFound, TranscriptsDisabled):
        try:
            # Fallback: any available language
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception:
            return (
                "ERROR: Transcript not available for this video. "
                "Captions may be disabled by the creator."
            )

    # Compact text (token-safe)
    return " ".join(item["text"] for item in transcript)

# -------------------------------------------------
# Agent Definition (STRICT & CONTROLLED)
# -------------------------------------------------
agent = Agent(
    name="YouTube Transcript Assistant",
    instructions=(
        "You are an academic YouTube assistant. "
        "ALWAYS call fetch_youtube_transcript when a YouTube link is present. "
        "If transcript text starts with 'ERROR:', politely explain the issue. "
        "If transcript exists, summarize clearly in simple bullet points. "
        "Focus on clarity, not length."
    ),
    tools=[fetch_youtube_transcript],
    model=model,
)

# ------------------------------
# Chainlit UI (ENHANCED)
# ------------------------------

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🎓 **YouTube Transcript Based AI Agent**\n"
            "_Programming Fundamentals Project_\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "### 📌 What this agent does\n"
            "- Extracts transcript from YouTube videos\n"
            "- Analyzes content using AI\n"
            "- Provides **clear summaries & key points**\n\n"
            "### 🧪 How to use\n"
            "1. Paste a YouTube video link\n"
            "2. Ask:\n"
            "   - *Summarize this video*\n"
            "   - *Explain main concepts*\n"
            "   - *What is this video about?*\n\n"
            "⚠️ _Works best with caption-enabled videos_\n\n"
            "---\n"
            "👨‍💻 **Developed by:** Sikandar Tahir  \n"
            "📘 **Subject:** Programming Fundamentals  \n"
            "🏫 **University Project Submission**\n"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    stream_msg = cl.Message(content="")
    await stream_msg.send()

    result = Runner.run_streamed(
        agent,
        [{"role": "user", "content": message.content}],
        run_config=config,
    )

    async for event in result.stream_events():
        if (
            event.type == "raw_response_event"
            and isinstance(event.data, ResponseTextDeltaEvent)
        ):
            await stream_msg.stream_token(event.data.delta)

        elif event.type == "run_item_stream_event":
          if event.item.type == "tool_call_item":
            await cl.Message(
            content="🔍 **Analyzing video and extracting transcript...**"
        ).send()


    await stream_msg.update()
