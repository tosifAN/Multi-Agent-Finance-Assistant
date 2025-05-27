# Multi-Agent Finance Assistant

A multi-source, multi-agent finance assistant that delivers spoken market briefs via a Streamlit app. This system provides portfolio managers with daily market insights, risk exposure analysis, and earnings surprises through a voice interface.

## Project Overview

This finance assistant integrates multiple specialized agents to deliver comprehensive market analysis:

- **API Agent**: Polls real-time & historical market data via AlphaVantage/Yahoo Finance
- **Scraping Agent**: Crawls financial filings using Python loaders
- **Retriever Agent**: Indexes embeddings in FAISS/Pinecone and retrieves relevant information
- **Analysis Agent**: Performs quantitative analysis on financial data
- **Language Agent**: Synthesizes narrative via LLM using LangChain's retriever interface
- **Voice Agent**: Handles Speech-to-Text (Whisper) → LLM → Text-to-Speech pipelines

## Architecture

### Microservices Architecture

The system is built using FastAPI microservices for each agent, with the following routing logic:

```
Voice input → STT → Orchestrator → RAG/Analysis → LLM → TTS or text
```

If retrieval confidence is below threshold, the system prompts user clarification via voice.

### Technology Stack

- **Data Ingestion**: APIs, web scraping, document loaders
- **Vector Store**: FAISS for embedding indexing and retrieval
- **Agent Frameworks**: LangGraph and CrewAI
- **Voice Processing**: Whisper (STT) and open-source TTS
- **Frontend**: Streamlit
- **API Gateway**: FastAPI

## Setup Instructions

### Prerequisites

- Python 3.9+
- Docker (optional for containerization)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables (API keys, etc.)
4. Run all the servers(agents)
5. Run the application:
   ```
   streamlit run streamlit_app/app.py
   ```

## Project Structure

```
/data_ingestion - Data collection modules (API, scraping)
/agents - Individual agent implementations
/orchestrator - Agent coordination and workflow management
/streamlit_app - User interface and deployment
/docs - Documentation including AI tool usage logs
```

## Performance Benchmarks

(To be added after implementation)

## Framework Comparisons

(To be added after implementation)

## License

Open Source - MIT