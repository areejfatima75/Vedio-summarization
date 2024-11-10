import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from transformers import pipeline

# Load and cache the summarization model
@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text_summary = load_summarization_model()

def summary(input_text):
    try:
        output = text_summary(input_text)
        if output and 'summary_text' in output[0]:
            return output[0]['summary_text']
        else:
            return "Summarization model returned an empty response."
    except Exception as e:
        return f"An error occurred during summarization: {e}"

def extract_video_id(url):
    # Regex to extract the video ID from various YouTube URL formats
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def get_youtube_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Video ID could not be extracted. Please check the URL and try again."

    try:
        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Format the transcript into plain text
        formatter = TextFormatter()
        text_transcript = formatter.format_transcript(transcript)
        
        # Summarize the transcript
        summary_text = summary(text_transcript)
        return summary_text
    except Exception as e:
        return f"An error occurred while fetching the transcript or summarizing: {e}"

# Streamlit UI
st.title("@GenAILearniverse Project 2: YouTube Script Summarizer")
st.write("This application summarizes YouTube video scripts. Paste a YouTube URL below to get the summary.")

video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    st.write("Fetching and summarizing the video...")
    result = get_youtube_transcript(video_url)
    st.subheader("Summarized Text:")
    st.write(result)
