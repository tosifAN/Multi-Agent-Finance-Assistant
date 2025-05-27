import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

# Import necessary modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define request and response models
class AnalysisRequest(BaseModel):
    query_type: str  # 'risk_exposure', 'earnings_analysis', 'market_sentiment', 'portfolio_metrics'
    data: Dict[str, Any]
    parameters: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str

# Create the FastAPI app
app = FastAPI(title="Analysis Agent", description="Agent for performing quantitative analysis on financial data")

class AnalysisAgent:
    """Agent for performing quantitative analysis on financial data"""
    
    def __init__(self):
        """Initialize the analysis agent"""
        pass
    
    async def analyze_risk_exposure(self, data: Dict, parameters: Optional[Dict] = None) -> Dict:
        """Analyze risk exposure for a portfolio
        
        Args:
            data: Dictionary containing portfolio and market data
            parameters: Optional parameters for the analysis
            
        Returns:
            Dictionary containing risk exposure metrics
        """
        # Extract data
        allocation_percentage = data.get("allocation_percentage", 0)
        previous_allocation = data.get("previous_allocation_percentage", 0)
        market_caps = data.get("market_caps", {})
        daily_changes = data.get("daily_changes", {})
        
        # Calculate concentration risk
        total_market_cap = sum(market_caps.values())
        concentration = {}
        for symbol, cap in market_caps.items():
            concentration[symbol] = (cap / total_market_cap) * 100 if total_market_cap else 0
        
        # Calculate volatility (simplified)
        volatility = np.std(list(daily_changes.values())) if daily_changes else 0
        
        # Calculate allocation change
        allocation_change = allocation_percentage - previous_allocation
        
        # Calculate risk metrics
        top_holdings = sorted(concentration.items(), key=lambda x: x[1], reverse=True)[:5]
        top_holdings_dict = {symbol: value for symbol, value in top_holdings}
        
        return {
            "allocation_percentage": allocation_percentage,
            "previous_allocation": previous_allocation,
            "allocation_change": allocation_change,
            "concentration_risk": {
                "top_holdings": top_holdings_dict,
                "concentration_score": sum(value**2 for value in concentration.values()) / 10000  # HHI index / 10000
            },
            "volatility": volatility,
            "risk_score": (volatility * 0.6) + (allocation_change * 0.4) if allocation_change > 0 else volatility * 0.8
        }
    
    async def analyze_earnings(self, data: Dict, parameters: Optional[Dict] = None) -> Dict:
        """Analyze earnings surprises
        
        Args:
            data: Dictionary containing earnings data
            parameters: Optional parameters for the analysis
            
        Returns:
            Dictionary containing earnings analysis
        """
        # Extract data
        surprises = data.get("surprises", [])
        
        # Calculate metrics
        positive_surprises = [s for s in surprises if s.get("surprise_percent", 0) > 0]
        negative_surprises = [s for s in surprises if s.get("surprise_percent", 0) < 0]
        
        # Calculate average surprise
        avg_surprise = sum(s.get("surprise_percent", 0) for s in surprises) / len(surprises) if surprises else 0
        
        # Format significant surprises
        significant_surprises = []
        for surprise in surprises:
            if abs(surprise.get("surprise_percent", 0)) >= 2:  # 2% threshold for significance
                significant_surprises.append({
                    "symbol": surprise.get("symbol", ""),
                    "name": surprise.get("name", ""),
                    "surprise_percent": surprise.get("surprise_percent", 0),
                    "direction": "beat" if surprise.get("surprise_percent", 0) > 0 else "missed"
                })
        
        return {
            "total_reports": len(surprises),
            "positive_surprises": len(positive_surprises),
            "negative_surprises": len(negative_surprises),
            "average_surprise": avg_surprise,
            "significant_surprises": significant_surprises
        }
    
    async def analyze_market_sentiment(self, data: Dict, parameters: Optional[Dict] = None) -> Dict:
        """Analyze market sentiment
        
        Args:
            data: Dictionary containing sentiment data
            parameters: Optional parameters for the analysis
            
        Returns:
            Dictionary containing sentiment analysis
        """
        # Extract data
        sentiment_score = data.get("sentiment_score", 0)
        overall_sentiment = data.get("overall_sentiment", "neutral")
        sentiment_counts = data.get("sentiment_counts", {})
        news_items = data.get("news_items", [])
        
        # Calculate sentiment trend (if we had historical data)
        # For now, we'll use a placeholder
        sentiment_trend = "stable"
        
        # Extract key themes from news
        themes = {}
        for item in news_items:
            summary = item.get("summary", "")
            # In a real implementation, we would use NLP to extract themes
            # For now, we'll use some simple keyword matching
            if "yield" in summary.lower() or "interest rate" in summary.lower():
                themes["interest_rates"] = themes.get("interest_rates", 0) + 1
            if "chip" in summary.lower() or "semiconductor" in summary.lower():
                themes["semiconductors"] = themes.get("semiconductors", 0) + 1
            if "regulation" in summary.lower() or "regulatory" in summary.lower():
                themes["regulation"] = themes.get("regulation", 0) + 1
        
        # Sort themes by frequency
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "sentiment_score": sentiment_score,
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_counts,
            "sentiment_trend": sentiment_trend,
            "key_themes": dict(sorted_themes)
        }
    
    async def calculate_portfolio_metrics(self, data: Dict, parameters: Optional[Dict] = None) -> Dict:
        """Calculate portfolio metrics
        
        Args:
            data: Dictionary containing portfolio data
            parameters: Optional parameters for the calculation
            
        Returns:
            Dictionary containing portfolio metrics
        """
        # Extract data
        allocation_percentage = data.get("allocation_percentage", 0)
        market_caps = data.get("market_caps", {})
        daily_changes = data.get("daily_changes", {})
        
        # Calculate metrics
        weighted_return = sum(daily_changes.get(symbol, 0) * (cap / sum(market_caps.values()) if sum(market_caps.values()) else 0)
                             for symbol, cap in market_caps.items())
        
        # Calculate diversification score (simplified)
        num_stocks = len(market_caps)
        diversification_score = min(1.0, num_stocks / 20)  # Max score at 20 stocks
        
        # Calculate exposure metrics
        total_market_cap = sum(market_caps.values())
        exposure = {symbol: (cap / total_market_cap) * 100 if total_market_cap else 0 for symbol, cap in market_caps.items()}
        
        return {
            "allocation_percentage": allocation_percentage,
            "weighted_return": weighted_return,
            "diversification_score": diversification_score,
            "exposure": exposure,
            "num_holdings": num_stocks
        }
    
    async def process_request(self, request: AnalysisRequest) -> Dict:
        """Process an analysis request
        
        Args:
            request: AnalysisRequest object
            
        Returns:
            Dictionary containing the response data
        """
        try:
            if request.query_type == "risk_exposure":
                data = await self.analyze_risk_exposure(request.data, request.parameters)
            elif request.query_type == "earnings_analysis":
                data = await self.analyze_earnings(request.data, request.parameters)
            elif request.query_type == "market_sentiment":
                data = await self.analyze_market_sentiment(request.data, request.parameters)
            elif request.query_type == "portfolio_metrics":
                data = await self.calculate_portfolio_metrics(request.data, request.parameters)
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

# Dependency to get the analysis agent
def get_analysis_agent():
    return AnalysisAgent()

# API routes
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest, 
                     analysis_agent: AnalysisAgent = Depends(get_analysis_agent)):
    """Endpoint for analyzing financial data"""
    response = await analysis_agent.process_request(request)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the FastAPI app if this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)