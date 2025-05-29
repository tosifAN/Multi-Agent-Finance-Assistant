# Multi-Agent Finance Assistant

A multi-source, multi-agent finance assistant that delivers spoken market briefs via a Streamlit app. This system implements advanced data-ingestion pipelines (APIs, web scraping, document loaders), indexes embeddings in a vector store for Retrieval-Augmented Generation (RAG), and orchestrates specialized agents via FastAPI microservices.

## Architecture

The system is built with a microservices architecture using FastAPI for each agent:

- **API Agent**: Polls real-time & historical market data via AlphaVantage or Yahoo Finance
- **Scraping Agent**: Crawls financial filings and news
- **Retriever Agent**: Indexes embeddings in FAISS and retrieves top-k chunks
- **Analysis Agent**: Performs financial analysis on retrieved data
- **Language Agent**: Synthesizes narrative via LLM using LangChain's retriever interface
- **Voice Agent**: Handles STT (Whisper) → LLM → TTS pipelines

## Project Structure

```
/
├── agents/                # Agent implementations
├── data_ingestion/        # Data ingestion pipelines
├── orchestrator/          # Agent orchestration logic
├── streamlit_app/         # Streamlit frontend
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
└── .env.example          # Environment variables template
```

## Setup Instructions

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your API keys
6. Start the services:
   - FastAPI services: `python -m orchestrator.main`
   - Streamlit app: `streamlit run streamlit_app/app.py`

## Framework & Toolkit Choices

- **Agent Framework**: CrewAI for agent orchestration
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: OpenAI for natural language processing
- **Voice Processing**: Whisper for STT, gTTS/pyttsx3 for TTS
- **Data Processing**: Alpha Vantage and Yahoo Finance APIs, BeautifulSoup for web scraping

## Performance Benchmarks

(To be added after implementation)

## License

Open Source