## This is backend(fastapi server)

# Multi-Agent Finance Assistant

A multi-source, multi-agent finance assistant that delivers spoken market briefs via a Streamlit app. This system implements advanced data-ingestion pipelines (APIs, web scraping, document loaders), indexes embeddings in a vector store for Retrieval-Augmented Generation (RAG), and orchestrates specialized agents via FastAPI microservices.

## Architecture

The system is built with a microservices architecture using FastAPI for each agent

| Agent Type        | Toolkits / Libraries                           |
|------------------|-------------------------------------------------|
| API Agent         | `yfinance`, `AlphaVantage`
| Scraping Agent    | `unstructured`, `playwright`, `bs4`, `requests-html` |
| Retriever Agent   | `Pinecone`, `OpenAIEmbeddings`, `InstructorEmbedding` |
| Analytics Agent   | `pandas`, `numpy`                              |
| Language Agent    | `LangChain`, `OpenAI GPT-4`, `CrewAI`          |
| Voice Agent       | `Whisper`, `pyttsx3`, `EdgeTTS`                |
| UI & Routing      | `Streamlit`, `FastAPI`, `reactJS`              |

---



## Framework & Toolkit Choices

- **Agent Framework**: CrewAI for agent orchestration
- **Vector Store**: PineCone for efficient similarity search
- **LLM**: OpenAI for natural language processing
- **Voice Processing**: Whisper for STT, gTTS/pyttsx3 for TTS
- **Data Processing**: Alpha Vantage and Yahoo Finance APIs, BeautifulSoup for web scraping

## 📊 Performance Benchmarks

### Latency Metrics
- **Voice-to-Voice**: < 3 seconds end-to-end
- **RAG Retrieval**: < 700ms for top-k search
- **Market Data**: < 200ms cached, < 2s fresh
- **TTS Generation**: < 3s per response

### Accuracy Metrics
- **STT Accuracy**: > 98% (financial terminology)
- **RAG Relevance**: > 78% (semantic similarity)
- **Market Data Freshness**: < 3min delay
- **Synthesis Quality**:  OpenAI

## Futual Goals
- Make it a MCP hosted in cloud server.
- Add some agents like DuckDuckGo for online search functionality.
- Wanna to add some remote MCP's.
- Hosted that cloud server that reduces the latency.


## Research
- Starting with the building using various agent library like phi, Langgraph and finally CrewAI
- Build the understanding on MCP's
- Build the whole project
