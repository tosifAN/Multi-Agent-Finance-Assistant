import os
import sys
import asyncio
import streamlit as st
import tempfile
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator.workflow import AgentOrchestrator

# Set page config
st.set_page_config(
    page_title="Finance Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator()

# Title and description
st.title("Multi-Agent Finance Assistant")
st.markdown("""
    This assistant provides market insights, risk exposure analysis, and earnings surprises for portfolio managers.
    You can interact with it using text or voice input.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This finance assistant integrates multiple specialized agents:
    
    - **API Agent**: Polls real-time & historical market data
    - **Scraping Agent**: Crawls financial filings
    - **Retriever Agent**: Indexes embeddings and retrieves information
    - **Analysis Agent**: Performs quantitative analysis
    - **Language Agent**: Synthesizes narrative via LLM
    - **Voice Agent**: Handles speech-to-text and text-to-speech
    """)
    
    st.header("Sample Queries")
    st.markdown("""
    - What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?
    - How has the semiconductor sector performed this week?
    - What are the key market themes affecting Asian tech stocks?
    """)

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # Input options
    input_type = st.radio("Choose input method:", ["Text", "Voice"], horizontal=True)
    
    if input_type == "Text":
        # Text input
        query = st.text_input("Enter your query:")
        submit_button = st.button("Submit")
        
        if submit_button and query:
            with st.spinner("Processing your query..."):
                # Process the query
                response = asyncio.run(st.session_state.orchestrator.process_query(query))
                
                # Add to history
                st.session_state.history.append({
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    else:
        # Voice input
        st.write("Click the button below and speak your query:")
        audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
        record_button = st.button("Record")
        
        if record_button:
            # In a real implementation, this would use the device microphone
            # For now, we'll simulate recording
            st.warning("Recording functionality requires microphone access. Please upload an audio file instead.")
        
        if audio_file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                audio_path = tmp_file.name
            
            with st.spinner("Processing your voice query..."):
                # Process the voice query
                response = asyncio.run(st.session_state.orchestrator.process_query(
                    "", is_voice=True, audio_file_path=audio_path
                ))
                
                # Add to history
                st.session_state.history.append({
                    "query": response.get("transcribed_query", "[Voice Input]"),
                    "response": response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "is_voice": True
                })
                
                # Play audio response if available
                if "audio_path" in response and response["audio_path"]:
                    st.audio(response["audio_path"])

# Display conversation history
st.header("Conversation History")
for item in reversed(st.session_state.history):
    with st.expander(f"{item['timestamp']} - {item['query'][:50]}..."):
        st.markdown(f"**Query:** {item['query']}")
        st.markdown(f"**Response:** {item['response']['text_response']}")
        
        # Display additional context if available
        if "context" in item["response"]:
            st.markdown("**Context:**")
            st.json(item["response"]["context"])

with col2:
    # Market Overview
    st.header("Market Overview")
    
    # In a real implementation, this would fetch real-time data
    # For demonstration, we'll use placeholder data
    st.metric("Asia Tech Allocation", "22%", "+4%")
    st.metric("TSMC", "$563.20", "+4.2%")
    st.metric("Samsung", "$1,245.30", "-2.1%")
    
    # Market Sentiment
    st.subheader("Market Sentiment")
    st.progress(0.6, "Neutral with cautionary tilt")
    
    # Key Themes
    st.subheader("Key Themes")
    themes = ["Rising Yields", "Semiconductor Demand", "Regulatory Changes"]
    for theme in themes:
        st.markdown(f"• {theme}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2023 Finance Assistant | Powered by AI Agents</p>
</div>
""", unsafe_allow_html=True)