import os
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

# Import the web scraper from data_ingestion
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_ingestion.web_scraper import WebScraperClient, FinancialFilingsScraper, SECFilingsScraper

# Define request and response models
class ScrapingRequest(BaseModel):
    query_type: str  # 'earnings_reports', 'market_news', 'sec_filings', 'sentiment_analysis'
    parameters: Optional[Dict[str, Any]] = None

class ScrapingResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str

# Create the FastAPI app
app = FastAPI(title="Scraping Agent", description="Agent for scraping financial filings and market news")

# Dependency to get the web scraper client
def get_web_scraper_client():
    return WebScraperClient()

class ScrapingAgent:
    """Agent for scraping financial filings and market news"""
    
    def __init__(self, web_scraper_client: Optional[WebScraperClient] = None):
        """Initialize the scraping agent
        
        Args:
            web_scraper_client: Optional WebScraperClient instance
        """
        self.web_scraper_client = web_scraper_client or WebScraperClient()
    
    async def get_earnings_reports(self, parameters: Optional[Dict] = None) -> Dict:
        """Get recent earnings reports
        
        Args:
            parameters: Optional parameters for the request
            
        Returns:
            Dictionary containing earnings report data
        """
        # Extract parameters
        days_back = parameters.get("days_back", 7) if parameters else 7
        
        # Get earnings reports from the web scraper
        reports = self.web_scraper_client.financial_filings.get_earnings_reports(days_back=days_back)
        
        # Format the response
        return {
            "reports": reports,
            "count": len(reports),
            "days_back": days_back
        }
    
    async def get_market_news(self, parameters: Optional[Dict] = None) -> Dict:
        """Get recent market news
        
        Args:
            parameters: Optional parameters for the request
            
        Returns:
            Dictionary containing market news data
        """
        # Extract parameters
        region = parameters.get("region", "asia") if parameters else "asia"
        category = parameters.get("category", "tech") if parameters else "tech"
        limit = parameters.get("limit", 5) if parameters else 5
        
        # Get market news from the web scraper
        news = self.web_scraper_client.financial_filings.get_market_news(
            region=region,
            category=category,
            limit=limit
        )
        
        # Format the response
        return {
            "news": news,
            "count": len(news),
            "region": region,
            "category": category
        }
    
    async def get_sec_filings(self, parameters: Dict) -> Dict:
        """Get SEC filings for a company
        
        Args:
            parameters: Parameters for the request including symbol and form_type
            
        Returns:
            Dictionary containing SEC filing data
        """
        # Extract parameters
        symbol = parameters.get("symbol")
        if not symbol:
            raise ValueError("'symbol' parameter is required for sec_filings query")
            
        form_type = parameters.get("form_type", "10-Q")
        limit = parameters.get("limit", 5)
        
        # Get SEC filings from the web scraper
        filings = self.web_scraper_client.sec_filings.get_recent_filings(
            symbol=symbol,
            form_type=form_type,
            limit=limit
        )
        
        # Format the response
        return {
            "filings": filings,
            "count": len(filings),
            "symbol": symbol,
            "form_type": form_type
        }
    
    async def get_sentiment_analysis(self, parameters: Optional[Dict] = None) -> Dict:
        """Get sentiment analysis for Asia tech stocks
        
        Args:
            parameters: Optional parameters for the request
            
        Returns:
            Dictionary containing sentiment analysis data
        """
        # Get sentiment analysis from the web scraper
        sentiment = self.web_scraper_client.get_asia_tech_sentiment()
        
        # Format the response
        return sentiment
    
    async def process_request(self, request: ScrapingRequest) -> Dict:
        """Process a scraping request
        
        Args:
            request: ScrapingRequest object
            
        Returns:
            Dictionary containing the response data
        """
        try:
            if request.query_type == "earnings_reports":
                data = await self.get_earnings_reports(request.parameters)
            elif request.query_type == "market_news":
                data = await self.get_market_news(request.parameters)
            elif request.query_type == "sec_filings":
                if not request.parameters or "symbol" not in request.parameters:
                    raise ValueError("'symbol' parameter is required for sec_filings query")
                data = await self.get_sec_filings(request.parameters)
            elif request.query_type == "sentiment_analysis":
                data = await self.get_sentiment_analysis(request.parameters)
            else:
                raise ValueError(f"Unsupported query type: {request.query_type}")
            
            return {
                "status": "success",
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }

# API routes
@app.post("/api/scraping", response_model=ScrapingResponse)
async def scrape_data(request: ScrapingRequest, 
                     web_scraper_client: WebScraperClient = Depends(get_web_scraper_client)):
    """Endpoint for scraping financial data"""
    agent = ScrapingAgent(web_scraper_client)
    response = await agent.process_request(request)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the FastAPI app if this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)