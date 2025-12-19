# app.py
import streamlit as st
import asyncio
from main import get_youtube_transcript, agent, Runner, config
from openai.types.responses import ResponseTextDeltaEvent

st.title("🎬 YouTube Transcript AI Agent")

video_url = st.text_input("YouTube Video URL")
user_question = st.text_input("Ask something about the video")

if st.button("Get Answer"):
    if not video_url or not user_question:
        st.warning("Provide both URL and question!")
    else:
        transcript = get_youtube_transcript(video_url)
        if transcript.startswith("ERROR"):
            st.error(transcript)
        else:
            st.info("Transcript fetched!")

            async def run_agent():
                input_items = [{"role": "user", "content": f"{user_question}\nTranscript:\n{transcript}"}]
                result = Runner.run_streamed(agent, input_items, run_config=config)
                response_text = ""
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        response_text += event.data.delta
                return response_text

            answer = asyncio.run(run_agent())
            st.success("AI Answer:")
            st.write(answer)
