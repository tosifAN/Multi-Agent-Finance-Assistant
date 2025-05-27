import os
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

# Import necessary modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")  # Replace with actual path


# Define request and response models
class LanguageRequest(BaseModel):
    query: str
    context: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None

class LanguageResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str

# Create the FastAPI app
app = FastAPI(title="Language Agent", description="Agent for synthesizing narratives via LLM using LangChain's retriever interface")

class LanguageAgent:
    """Agent for synthesizing narratives via LLM"""
    
    def __init__(self):
        """Initialize the language agent"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
           print("OPENAI_API_KEY environment variable is not set!")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            from langchain.llms import OpenAI
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
        except ImportError:
            raise ImportError("Please install langchain: pip install langchain")
        
        self.llm = OpenAI(temperature=0.7, openai_api_key=self.openai_api_key)
    
    async def generate_market_brief(self, query: str, context: Dict, parameters: Optional[Dict] = None) -> Dict:
        """Generate a market brief based on the query and context
        
        Args:
            query: User query
            context: Dictionary containing context information
            parameters: Optional parameters for the generation
            
        Returns:
            Dictionary containing the generated brief
        """
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain
        except ImportError:
            raise ImportError("Please install langchain: pip install langchain")
        
        # Extract context data
        risk_exposure = context.get("risk_exposure", {})
        earnings_analysis = context.get("earnings_analysis", {})
        market_sentiment = context.get("market_sentiment", {})
        
        # Create prompt template
        template = """
        You are a financial advisor providing a morning market brief to a portfolio manager.
        
        The portfolio manager asked: "{query}"
        
        Based on the following information, provide a concise and informative response:
        
        Risk Exposure:
        - Asia tech allocation: {allocation_percentage}% of AUM (previously {previous_allocation}%)
        - Allocation change: {allocation_change}%
        - Top holdings: {top_holdings}
        - Volatility: {volatility}
        
        Earnings Analysis:
        - Significant surprises: {significant_surprises}
        
        Market Sentiment:
        - Overall sentiment: {overall_sentiment}
        - Key themes: {key_themes}
        
        Your response should be professional, concise, and directly address the query.
        Focus on the most important insights and actionable information.
        """
        
        # Format significant surprises for the prompt
        significant_surprises = earnings_analysis.get("significant_surprises", [])
        surprises_text = ""
        for surprise in significant_surprises:
            direction = surprise.get("direction", "")
            symbol = surprise.get("symbol", "")
            name = surprise.get("name", "")
            percent = surprise.get("surprise_percent", 0)
            surprises_text += f"{name} ({symbol}) {direction} estimates by {abs(percent)}%, "
        
        if not surprises_text:
            surprises_text = "No significant earnings surprises"
        
        # Format top holdings for the prompt
        top_holdings = risk_exposure.get("concentration_risk", {}).get("top_holdings", {})
        holdings_text = ", ".join([f"{symbol}: {value:.1f}%" for symbol, value in top_holdings.items()])
        
        # Format key themes for the prompt
        key_themes = market_sentiment.get("key_themes", {})
        themes_text = ", ".join([f"{theme}" for theme in key_themes.keys()])
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["query", "allocation_percentage", "previous_allocation", "allocation_change", 
                           "top_holdings", "volatility", "significant_surprises", "overall_sentiment", "key_themes"],
            template=template
        )
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Run chain
        response = chain.run(
            query=query,
            allocation_percentage=risk_exposure.get("allocation_percentage", 0),
            previous_allocation=risk_exposure.get("previous_allocation", 0),
            allocation_change=risk_exposure.get("allocation_change", 0),
            top_holdings=holdings_text,
            volatility=f"{risk_exposure.get('volatility', 0):.2f}",
            significant_surprises=surprises_text,
            overall_sentiment=market_sentiment.get("overall_sentiment", "neutral"),
            key_themes=themes_text
        )
        
        return {
            "brief": response.strip(),
            "query": query
        }
    
    async def generate_response(self, query: str, context: Dict, parameters: Optional[Dict] = None) -> Dict:
        """Generate a response based on the query and context
        
        Args:
            query: User query
            context: Dictionary containing context information
            parameters: Optional parameters for the generation
            
        Returns:
            Dictionary containing the generated response
        """
        # For now, we'll just use the market brief generation
        # In a more complex implementation, we could have different response types
        return await self.generate_market_brief(query, context, parameters)
    
    async def process_request(self, request: LanguageRequest) -> Dict:
        """Process a language request
        
        Args:
            request: LanguageRequest object
            
        Returns:
            Dictionary containing the response data
        """
        try:
            # Generate response
            response = await self.generate_response(
                query=request.query,
                context=request.context,
                parameters=request.parameters
            )
            
            return {
                "status": "success",
                "data": response,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }

# Dependency to get the language agent
def get_language_agent():
    return LanguageAgent()

# API routes
@app.post("/api/generate", response_model=LanguageResponse)
async def generate_language(request: LanguageRequest, 
                         language_agent: LanguageAgent = Depends(get_language_agent)):
    """Endpoint for generating language responses"""
    response = await language_agent.process_request(request)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the FastAPI app if this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)