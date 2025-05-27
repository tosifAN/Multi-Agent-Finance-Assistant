# AI Tool Usage Log

This document provides a detailed log of AI tool usage during the development of the Multi-Agent Finance Assistant project. It includes prompts, code generation steps, and model parameters used throughout the development process.

## Project Setup

### Initial Project Structure

AI was used to create the initial project structure based on the assignment requirements. The following components were generated:

- README.md with project overview and setup instructions
- requirements.txt with necessary dependencies
- .env.example file for environment variables
- Directory structure for data_ingestion, agents, orchestrator, and streamlit_app

### Code Generation

The following code modules were generated using AI assistance:

#### Data Ingestion

- **api_client.py**: Implemented clients for AlphaVantage and Yahoo Finance APIs
- **web_scraper.py**: Created web scrapers for financial filings and market news
- **document_loader.py**: Developed document loaders for various file formats (PDF, DOCX, CSV, etc.)

#### Agent Implementations

- **api_agent.py**: FastAPI service for fetching market data
- **scraping_agent.py**: FastAPI service for web scraping
- **retriever_agent.py**: FastAPI service for vector store retrieval with FAISS/Pinecone
- **analysis_agent.py**: FastAPI service for quantitative analysis
- **language_agent.py**: FastAPI service for narrative generation using LangChain
- **voice_agent.py**: FastAPI service for speech-to-text and text-to-speech

#### Orchestration

- **workflow.py**: Implemented the orchestrator to coordinate all agents

#### User Interface

- **app.py**: Created Streamlit app for user interaction

## AI Models and Parameters

### Language Models

- **OpenAI GPT**: Used for narrative generation in the Language Agent
  - Temperature: 0.7
  - Model: Default GPT model via LangChain

### Embedding Models

- **Sentence Transformers**: Used for document embedding in the Retriever Agent
  - Model: 'all-MiniLM-L6-v2'
  - Dimension: 384

### Speech Models

- **Whisper**: Used for speech-to-text conversion
  - Model: 'base'
- **pyttsx3**: Used for text-to-speech conversion
  - Voice: 'en-US-Neural2-F'
  - Rate: 150
  - Volume: 1.0

## Development Process

### Prompt Engineering

The following prompts were used to guide the AI in generating specific components:

1. **Initial Project Structure**: "Create a multi-agent finance assistant that delivers spoken market briefs via a Streamlit app, following the requirements in the assignment."

2. **API Client**: "Implement API clients for fetching market data from AlphaVantage and Yahoo Finance."

3. **Web Scraper**: "Create web scrapers for financial filings and market news."

4. **Vector Store**: "Implement FAISS and Pinecone vector stores for document retrieval."

5. **Agent Implementations**: "Create FastAPI microservices for each specialized agent."

6. **Orchestrator**: "Implement an orchestrator to coordinate all agents and handle the workflow."

7. **Streamlit App**: "Create a user interface using Streamlit for interacting with the finance assistant."

### Iterative Development

The development process involved several iterations:

1. **Initial Framework**: Set up the basic project structure and dependencies
2. **Data Ingestion**: Implemented API clients and web scrapers
3. **Agent Implementation**: Created individual agent microservices
4. **Orchestration**: Developed the workflow for coordinating agents
5. **User Interface**: Built the Streamlit app for user interaction

## Challenges and Solutions

### Vector Store Implementation

- **Challenge**: Implementing efficient vector storage and retrieval
- **Solution**: Used FAISS for local development and provided Pinecone as an alternative for production

### Speech Processing

- **Challenge**: Implementing reliable speech-to-text and text-to-speech
- **Solution**: Used Whisper for STT and pyttsx3 for TTS with configurable parameters

### Agent Coordination

- **Challenge**: Coordinating multiple agents efficiently
- **Solution**: Implemented an orchestrator with async functions for parallel processing

## Future Improvements

- Implement more sophisticated NLP for theme extraction
- Add more comprehensive error handling and fallback mechanisms
- Improve the voice interface with better audio quality and more natural speech
- Implement more advanced RAG techniques for better retrieval accuracy