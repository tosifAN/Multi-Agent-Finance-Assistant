import os
import tempfile
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Crew, Process

# Import the API clients from data_ingestion
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agents
from agents.api_agent import APIAgent, create_api_tasks
from agents.scraping_agent import ScrapingAgent, create_scraping_tasks
from agents.retriever_agent import RetrieverAgent, create_retriever_tasks
from agents.analysis_agent import AnalysisAgent, create_analysis_tasks
from agents.language_agent import LanguageAgent, create_language_tasks
from agents.voice_agent import VoiceAgent, create_voice_tasks

# Create FastAPI app
app = FastAPI(title="Finance Assistant API", description="API for the multi-agent finance assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize agents
api_agent = APIAgent()
api_agent_instance = api_agent.create_agent()

scraping_agent = ScrapingAgent()
scraping_agent_instance = scraping_agent.create_agent()

retriever_agent = RetrieverAgent()
retriever_agent_instance = retriever_agent.create_agent()

analysis_agent = AnalysisAgent()
analysis_agent_instance = analysis_agent.create_agent()

language_agent = LanguageAgent()
language_agent_instance = language_agent.create_agent()

voice_agent = VoiceAgent()
voice_agent_instance = voice_agent.create_agent()

# Create tasks for each agent
api_tasks = create_api_tasks(api_agent_instance)
scraping_tasks = create_scraping_tasks(scraping_agent_instance)
retriever_tasks = create_retriever_tasks(retriever_agent_instance)
analysis_tasks = create_analysis_tasks(analysis_agent_instance)
language_tasks = create_language_tasks(language_agent_instance)
voice_tasks = create_voice_tasks(voice_agent_instance)

# Create the crew
finance_crew = Crew(
    agents=[api_agent_instance, scraping_agent_instance, retriever_agent_instance, 
            analysis_agent_instance, language_agent_instance, voice_agent_instance],
    tasks=api_tasks + scraping_tasks + retriever_tasks + analysis_tasks + language_tasks + voice_tasks,
    verbose=True,
    process=Process.sequential  # Use sequential process for predictable execution
)

# Define request and response models
class TextQuery(BaseModel):
    query: str

class MarketBriefResponse(BaseModel):
    brief: str
    audio_url: Optional[str] = None
    portfolio_data: Dict[str, Any]
    stock_performance: Dict[str, Any]
    earnings_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    audio_url: Optional[str] = None
    confidence: float
    sources: List[Dict[str, Any]]

# Background task for data collection and indexing
def collect_and_index_data():
    """Background task to collect and index financial data."""
    # Collect data from APIs
    asia_tech_stocks = api_agent.get_asia_tech_stocks()
    earnings_surprises = api_agent.get_earnings_surprises()
    portfolio_data = api_agent.calculate_asia_tech_exposure()
    
    # Collect data from web scraping
    financial_news = scraping_agent.scrape_financial_news()
    market_sentiment = scraping_agent.scrape_market_sentiment()
    
    # Index data in vector store
    retriever_agent.index_financial_data(asia_tech_stocks, 'stock_data')
    retriever_agent.index_financial_data(earnings_surprises, 'earnings')
    retriever_agent.index_financial_data(financial_news, 'news')
    retriever_agent.index_financial_data([market_sentiment], 'sentiment')
    
    return {
        'asia_tech_stocks': asia_tech_stocks,
        'earnings_surprises': earnings_surprises,
        'portfolio_data': portfolio_data,
        'financial_news': financial_news,
        'market_sentiment': market_sentiment
    }

# API endpoints
@app.get("/")
async def root():
    return {"message": "Finance Assistant API is running"}

@app.post("/market-brief", response_model=MarketBriefResponse)
async def get_market_brief(background_tasks: BackgroundTasks, voice_output: bool = Query(True)):
    """Generate a morning market brief for Asia tech stocks."""
    try:
        # Collect and analyze data
        data = collect_and_index_data()
        
        # Analyze the data
        stock_performance = analysis_agent.analyze_stock_performance(data['asia_tech_stocks'])
        earnings_analysis = analysis_agent.analyze_earnings_surprises(data['earnings_surprises'])
        sentiment_analysis = analysis_agent.analyze_market_sentiment(data['market_sentiment'], data['financial_news'])
        risk_analysis = analysis_agent.analyze_risk_exposure(data['portfolio_data'], data['asia_tech_stocks'], sentiment_analysis)
        
        # Generate market brief
        brief = language_agent.create_market_brief(
            data['portfolio_data'],
            stock_performance,
            earnings_analysis,
            sentiment_analysis,
            risk_analysis
        )
        
        # Generate audio if requested
        audio_url = None
        if voice_output:
            # Create a temporary file for the audio
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, 'market_brief.mp3')
            
            # Convert text to speech
            tts_result = voice_agent.text_to_speech(brief, audio_file)
            if tts_result['success']:
                audio_url = f"/audio/{os.path.basename(audio_file)}"
        
        return MarketBriefResponse(
            brief=brief,
            audio_url=audio_url,
            portfolio_data=data['portfolio_data'],
            stock_performance=stock_performance,
            earnings_analysis=earnings_analysis,
            sentiment_analysis=sentiment_analysis,
            risk_analysis=risk_analysis
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def answer_query(query: TextQuery, voice_output: bool = Query(True)):
    """Answer a specific query about Asia tech stocks."""
    try:
        # Retrieve relevant information
        retrieved_info = retriever_agent.retrieve_asia_tech_info(query.query)
        
        # Check confidence level
        if retrieved_info['below_threshold']:
            answer = f"I'm not confident in my answer (confidence: {retrieved_info['avg_confidence']:.1f}%). Could you please clarify your question?"
            confidence = retrieved_info['avg_confidence']
        else:
            # Generate answer
            answer = language_agent.answer_specific_query(query.query, retrieved_info['results'])
            confidence = retrieved_info['avg_confidence']
        
        # Generate audio if requested
        audio_url = None
        if voice_output:
            # Create a temporary file for the audio
            temp_dir = tempfile.gettempdir()
            audio_file = os.path.join(temp_dir, 'query_response.mp3')
            
            # Convert text to speech
            tts_result = voice_agent.text_to_speech(answer, audio_file)
            if tts_result['success']:
                audio_url = f"/audio/{os.path.basename(audio_file)}"
        
        return QueryResponse(
            answer=answer,
            audio_url=audio_url,
            confidence=confidence,
            sources=[{
                'content': r['content'],
                'metadata': r['metadata'],
                'confidence': r['confidence']
            } for r in retrieved_info['results']]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice-query")
async def process_voice_query(file: UploadFile = File(...), voice_output: bool = Query(True)):
    """Process a voice query and return the response."""
    try:
        # Save the uploaded audio file
        temp_dir = tempfile.gettempdir()
        audio_file = os.path.join(temp_dir, file.filename)
        with open(audio_file, "wb") as f:
            f.write(await file.read())
        
        # Transcribe the audio
        query_text, transcription = voice_agent.process_voice_query(audio_file)
        print(f"this is audio to text that user have gave ${query_text}")
        
        if not transcription['success']:
            raise HTTPException(status_code=400, detail="Failed to transcribe audio")
        
        # Process the query
        query_model = TextQuery(query=query_text)
        response = await answer_query(query_model, voice_output)
        
        # Add transcription to response
        response_dict = response.dict()
        response_dict['transcription'] = query_text
        
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files."""
    temp_dir = tempfile.gettempdir()
    audio_file = os.path.join(temp_dir, filename)
    
    if not os.path.exists(audio_file):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(audio_file, media_type="audio/mpeg")

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup."""
    # Start background data collection
    background_tasks = BackgroundTasks()
    background_tasks.add_task(collect_and_index_data)

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    host = os.getenv('FASTAPI_HOST', '0.0.0.0')
    port = int(os.getenv('FASTAPI_PORT', 8000))
    uvicorn.run("orchestrator.main:app", host=host, port=port, reload=True)