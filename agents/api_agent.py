import os
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime


# Import the API clients from data_ingestion
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_ingestion.api_client import MarketDataClient, AlphaVantageClient, YahooFinanceClient

# Define request and response models
class MarketDataRequest(BaseModel):
    query_type: str  # 'asia_tech_exposure', 'earnings_surprises', 'stock_data'
    parameters: Optional[Dict[str, Any]] = None

class MarketDataResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str

# Create the FastAPI app
app = FastAPI(title="API Agent", description="Agent for fetching market data from financial APIs")

# Dependency to get the market data client
def get_market_data_client():
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if alpha_vantage_key is None:
        print("ALPHA_VANTAGE_API_KEY environment variable is not set!")
    return MarketDataClient(alpha_vantage_key=alpha_vantage_key)

class APIAgent:
    """Agent for fetching market data from financial APIs"""
    
    def __init__(self, market_data_client: Optional[MarketDataClient] = None):
        """Initialize the API agent
        
        Args:
            market_data_client: Optional MarketDataClient instance
        """
        self.market_data_client = market_data_client or MarketDataClient()
    
    async def get_asia_tech_exposure(self, parameters: Optional[Dict] = None) -> Dict:
        """Get exposure to Asia tech stocks
        
        Args:
            parameters: Optional parameters for the request
            
        Returns:
            Dictionary containing exposure metrics
        """
        # Get exposure data from the market data client
        exposure_data = self.market_data_client.get_asia_tech_exposure()
        
        # Calculate allocation percentage (this would typically use portfolio data)
        # For demonstration, we'll use a sample allocation
        sample_aum = 1000000000  # $1 billion AUM
        asia_tech_allocation = exposure_data["total_market_cap"] * 0.22  # 22% allocation
        yesterday_allocation = 0.18  # 18% yesterday
        
        # Format the response
        return {
            "allocation_percentage": 22,
            "previous_allocation_percentage": 18,
            "allocation_change": 4,
            "total_market_cap": exposure_data["total_market_cap"],
            "daily_weighted_change": exposure_data["daily_weighted_change"],
            "stock_data": {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                          for k, v in exposure_data["stock_data"].items()},
            "market_caps": exposure_data["market_caps"],
            "daily_changes": exposure_data["daily_changes"]
        }
    
    async def get_earnings_surprises(self, parameters: Optional[Dict] = None) -> Dict:
        """Get recent earnings surprises
        
        Args:
            parameters: Optional parameters for the request
            
        Returns:
            Dictionary containing earnings surprise data
        """
        # Get earnings data from the market data client
        days_back = parameters.get("days_back", 7) if parameters else 7
        earnings_data = self.market_data_client.get_earnings_surprises(days_back=days_back)
        
        # Format the response
        return {
            "surprises": earnings_data,
            "count": len(earnings_data)
        }
    
    async def get_stock_data(self, parameters: Dict) -> Dict:
        """Get stock data for specific symbols
        
        Args:
            parameters: Parameters for the request including symbols, period, and interval
            
        Returns:
            Dictionary containing stock data
        """
        # Extract parameters
        symbols = parameters.get("symbols", [])
        period = parameters.get("period", "1mo")
        interval = parameters.get("interval", "1d")
        
        # Get stock data from Yahoo Finance
        stock_data = self.market_data_client.yahoo_finance.get_multiple_stocks_data(
            symbols=symbols,
            period=period,
            interval=interval
        )
        
        # Convert DataFrames to dictionaries for JSON serialization
        formatted_data = {}
        for symbol, df in stock_data.items():
            formatted_data[symbol] = df.to_dict(orient="records")
        
        return {
            "stock_data": formatted_data,
            "period": period,
            "interval": interval
        }
    
    async def process_request(self, request: MarketDataRequest) -> Dict:
        """Process a market data request
        
        Args:
            request: MarketDataRequest object
            
        Returns:
            Dictionary containing the response data
        """
        try:
            if request.query_type == "asia_tech_exposure":
                data = await self.get_asia_tech_exposure(request.parameters)
            elif request.query_type == "earnings_surprises":
                data = await self.get_earnings_surprises(request.parameters)
            elif request.query_type == "stock_data":
                if not request.parameters or "symbols" not in request.parameters:
                    raise ValueError("'symbols' parameter is required for stock_data query")
                data = await self.get_stock_data(request.parameters)
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
@app.post("/api/market-data", response_model=MarketDataResponse)
async def get_market_data(request: MarketDataRequest, 
                         market_data_client: MarketDataClient = Depends(get_market_data_client)):
    """Endpoint for fetching market data"""
    agent = APIAgent(market_data_client)
    response = await agent.process_request(request)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the FastAPI app if this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)