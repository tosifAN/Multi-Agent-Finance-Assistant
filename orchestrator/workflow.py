import os
import json
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrator for coordinating multiple agents"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the orchestrator
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default service URLs
        self.service_urls = {
            "api_agent": "http://localhost:8001",
            "scraping_agent": "http://localhost:8002",
            "retriever_agent": "http://localhost:8003",
            "analysis_agent": "http://localhost:8004",
            "language_agent": "http://localhost:8005",
            "voice_agent": "http://localhost:8006"
        }
        
        # Override with config if provided
        if "service_urls" in self.config:
            self.service_urls.update(self.config["service_urls"])
        
        # Confidence threshold for retrieval
        self.retrieval_confidence_threshold = self.config.get("retrieval_confidence_threshold", 0.7)
    
    async def call_agent_service(self, agent_name: str, endpoint: str, data: Dict) -> Dict:
        """Call an agent service
        
        Args:
            agent_name: Name of the agent service
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data from the agent service
        """
        if agent_name not in self.service_urls:
            raise ValueError(f"Unknown agent service: {agent_name}")
        
        url = f"{self.service_urls[agent_name]}{endpoint}"
        
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {agent_name} service: {e}")
            return {"status": "error", "data": {"error": str(e)}, "timestamp": datetime.now().isoformat()}
    
    async def process_voice_input(self, audio_file_path: str) -> str:
        """Process voice input to text
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        # Call the voice agent to convert speech to text
        response = await self.call_agent_service(
            agent_name="voice_agent",
            endpoint="/api/stt",
            data={"audio_file_path": audio_file_path}
        )
        
        if response["status"] == "success":
            return response["data"]["text"]
        else:
            raise Exception(f"Speech-to-text conversion failed: {response['data'].get('error', 'Unknown error')}")
    
    async def process_text_to_voice(self, text: str, voice: str = "en-US-Neural2-F") -> str:
        """Process text to voice
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            
        Returns:
            Path to the generated audio file
        """
        # Call the voice agent to convert text to speech
        response = await self.call_agent_service(
            agent_name="voice_agent",
            endpoint="/api/tts",
            data={"text": text, "voice": voice}
        )
        
        if response["status"] == "success":
            return response["data"]["audio_file_path"]
        else:
            raise Exception(f"Text-to-speech conversion failed: {response['data'].get('error', 'Unknown error')}")
    
    async def get_market_data(self, query_type: str, parameters: Optional[Dict] = None) -> Dict:
        """Get market data from the API agent
        
        Args:
            query_type: Type of market data query
            parameters: Optional parameters for the query
            
        Returns:
            Market data
        """
        # Call the API agent to get market data
        response = await self.call_agent_service(
            agent_name="api_agent",
            endpoint="/api/market-data",
            data={"query_type": query_type, "parameters": parameters or {}}
        )
        
        if response["status"] == "success":
            return response["data"]
        else:
            raise Exception(f"Market data query failed: {response['data'].get('error', 'Unknown error')}")
    
    async def get_scraped_data(self, query_type: str, parameters: Optional[Dict] = None) -> Dict:
        """Get scraped data from the scraping agent
        
        Args:
            query_type: Type of scraping query
            parameters: Optional parameters for the query
            
        Returns:
            Scraped data
        """
        # Call the scraping agent to get scraped data
        response = await self.call_agent_service(
            agent_name="scraping_agent",
            endpoint="/api/scraping",
            data={"query_type": query_type, "parameters": parameters or {}}
        )
        
        if response["status"] == "success":
            return response["data"]
        else:
            raise Exception(f"Scraping query failed: {response['data'].get('error', 'Unknown error')}")
    
    async def retrieve_information(self, query: str, collection: str, top_k: int = 5) -> Dict:
        """Retrieve information from the retriever agent
        
        Args:
            query: Query string
            collection: Collection to search
            top_k: Number of results to return
            
        Returns:
            Retrieved information
        """
        # Call the retriever agent to get information
        response = await self.call_agent_service(
            agent_name="retriever_agent",
            endpoint="/api/retrieve",
            data={"query": query, "collection": collection, "top_k": top_k}
        )
        
        if response["status"] == "success":
            return response["data"]
        else:
            raise Exception(f"Retrieval query failed: {response['data'].get('error', 'Unknown error')}")
    
    async def analyze_data(self, query_type: str, data: Dict, parameters: Optional[Dict] = None) -> Dict:
        """Analyze data using the analysis agent
        
        Args:
            query_type: Type of analysis
            data: Data to analyze
            parameters: Optional parameters for the analysis
            
        Returns:
            Analysis results
        """
        # Call the analysis agent to analyze data
        response = await self.call_agent_service(
            agent_name="analysis_agent",
            endpoint="/api/analyze",
            data={"query_type": query_type, "data": data, "parameters": parameters or {}}
        )
        
        if response["status"] == "success":
            return response["data"]
        else:
            raise Exception(f"Analysis failed: {response['data'].get('error', 'Unknown error')}")
    
    async def generate_narrative(self, query: str, context: Dict) -> Dict:
        """Generate a narrative using the language agent
        
        Args:
            query: User query
            context: Context information
            
        Returns:
            Generated narrative
        """
        # Call the language agent to generate a narrative
        response = await self.call_agent_service(
            agent_name="language_agent",
            endpoint="/api/generate",
            data={"query": query, "context": context}
        )
        
        if response["status"] == "success":
            return response["data"]
        else:
            raise Exception(f"Narrative generation failed: {response['data'].get('error', 'Unknown error')}")
    
    async def process_query(self, query: str, is_voice: bool = False, audio_file_path: Optional[str] = None) -> Dict:
        """Process a user query
        
        Args:
            query: User query (if text input)
            is_voice: Whether the input is voice
            audio_file_path: Path to the audio file (if voice input)
            
        Returns:
            Response data including text and optional audio path
        """
        try:
            # If voice input, convert to text
            if is_voice and audio_file_path:
                query = await self.process_voice_input(audio_file_path)
                logger.info(f"Transcribed query: {query}")
            
            # Process the query
            # For the morning market brief use case
            if "risk exposure" in query.lower() and "asia tech" in query.lower() and "earnings" in query.lower():
                # Get market data for Asia tech stocks
                asia_tech_data = await self.get_market_data("asia_tech_exposure")
                
                # Get earnings surprises
                earnings_data = await self.get_market_data("earnings_surprises")
                
                # Get market sentiment
                sentiment_data = await self.get_scraped_data("sentiment_analysis")
                
                # Analyze risk exposure
                risk_analysis = await self.analyze_data("risk_exposure", asia_tech_data)
                
                # Analyze earnings
                earnings_analysis = await self.analyze_data("earnings_analysis", {"surprises": earnings_data["surprises"]})
                
                # Analyze market sentiment
                sentiment_analysis = await self.analyze_data("market_sentiment", sentiment_data)
                
                # Generate narrative
                context = {
                    "risk_exposure": risk_analysis,
                    "earnings_analysis": earnings_analysis,
                    "market_sentiment": sentiment_analysis
                }
                
                narrative = await self.generate_narrative(query, context)
                
                # Convert to speech if voice input
                audio_path = None
                if is_voice:
                    audio_path = await self.process_text_to_voice(narrative["brief"])
                
                return {
                    "text_response": narrative["brief"],
                    "audio_path": audio_path,
                    "context": context
                }
            else:
                # For other queries, use retrieval-based approach
                # Retrieve relevant information
                retrieval_results = await self.retrieve_information(query, "market_news", top_k=3)
                
                # Check confidence
                if retrieval_results["confidence"] < self.retrieval_confidence_threshold:
                    # If confidence is low, prompt for clarification
                    clarification_text = "I'm not sure I understand your query. Could you please provide more details about what you're looking for?"
                    
                    # Convert to speech if voice input
                    audio_path = None
                    if is_voice:
                        audio_path = await self.process_text_to_voice(clarification_text)
                    
                    return {
                        "text_response": clarification_text,
                        "audio_path": audio_path,
                        "requires_clarification": True
                    }
                
                # Generate context from retrieval results
                context = {
                    "retrieved_information": retrieval_results["results"]
                }
                
                # Generate narrative
                narrative = await self.generate_narrative(query, context)
                
                # Convert to speech if voice input
                audio_path = None
                if is_voice:
                    audio_path = await self.process_text_to_voice(narrative["brief"])
                
                return {
                    "text_response": narrative["brief"],
                    "audio_path": audio_path,
                    "context": context
                }
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_message = f"Sorry, I encountered an error while processing your request: {str(e)}"
            
            # Convert to speech if voice input
            audio_path = None
            if is_voice:
                try:
                    audio_path = await self.process_text_to_voice(error_message)
                except Exception as tts_error:
                    logger.error(f"Error converting error message to speech: {tts_error}")
            
            return {
                "text_response": error_message,
                "audio_path": audio_path,
                "error": str(e)
            }

# Example usage
async def main():
    orchestrator = AgentOrchestrator()
    response = await orchestrator.process_query(
        "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    )
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(main())