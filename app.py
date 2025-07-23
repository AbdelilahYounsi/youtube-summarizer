
import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
from youtube_transcript_api import YouTubeTranscriptApi
os.environ["GEMINI_API_KEY"] = ""

# Set your API key as an environment variable
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"



st.title("YouTube Video Summarizer with CrewAI")

url = st.text_input("Enter a YouTube URL to summarize:")

if url:
    try:
        video_id = url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript])
        llm = LLM(
            model="gemini/gemini-2.5-pro",
            temperature=0.7,
        )
        # Define the agents
        researcher = Agent(
            role='Researcher',
            goal=f'Analyze the provided transcript and extract key information.',
            backstory="You are a world-class researcher, skilled at analyzing text and extracting the most relevant information.",
            verbose=False,
            allow_delegation=False,
            llm=llm
        )

        writer = Agent(
            role='Writer',
            goal='Write a concise and engaging summary of the provided transcript.',
            backstory="You are an expert writer, known for your ability to craft clear and compelling summaries.",
            verbose=False,
            allow_delegation=False,
            llm=llm
        )

        # Define the tasks
        research_task = Task(
            description=f"Analyze the following transcript and identify the main points: {transcript_text}",
            agent=researcher,
            expected_output="A detailed report with the main points and key takeaways from the transcript."
        )

        write_task = Task(
            description="Based on the research report, write a one-paragraph summary of the video.",
            agent=writer,
            expected_output="A concise and easy-to-read summary of the video's content."
        )

        # Create and run the crew
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            verbose=False
        )

        with st.spinner("Summarizing..."):
            result = crew.kickoff()
            st.subheader("Summary:")
            st.write(result)

    except Exception as e:
        st.error(f"An error occurred: {e}")
