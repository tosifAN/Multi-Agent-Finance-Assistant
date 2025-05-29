import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from crewai import Agent, Task

class AnalysisAgent:
    """Agent for performing financial analysis on market data."""
    
    def __init__(self):
        """Initialize the analysis agent."""
        pass
        
    def create_agent(self) -> Agent:
        """Create a CrewAI agent for financial analysis operations."""
        return Agent(
            role="Financial Analysis Specialist",
            goal="Analyze financial data to extract meaningful insights and trends",
            backstory="""You are an expert financial analyst with years of experience in 
            evaluating market data, earnings reports, and economic indicators. Your 
            specialty is in identifying patterns and insights that help investors make 
            informed decisions about their portfolios.""",
            verbose=True,
            allow_delegation=False
        )
    
    def analyze_stock_performance(self, stock_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance of stocks based on provided data.
        
        Args:
            stock_data: List of dictionaries with stock data
            
        Returns:
            Dictionary with analysis results
        """
        if not stock_data:
            return {
                'performance_summary': 'No data available for analysis',
                'top_performers': [],
                'bottom_performers': [],
                'average_change': None
            }
        
        # Extract change percentages
        changes = []
        for stock in stock_data:
            if 'change_pct' in stock and stock['change_pct'] is not None:
                changes.append({
                    'symbol': stock['symbol'],
                    'name': stock.get('name', stock['symbol']),
                    'change_pct': stock['change_pct']
                })
        
        # Sort by change percentage
        changes.sort(key=lambda x: x['change_pct'], reverse=True)
        
        # Calculate average change
        avg_change = sum(item['change_pct'] for item in changes) / len(changes) if changes else 0
        
        # Determine overall market direction
        if avg_change > 1.0:
            market_direction = 'strongly positive'
        elif avg_change > 0.2:
            market_direction = 'positive'
        elif avg_change > -0.2:
            market_direction = 'neutral'
        elif avg_change > -1.0:
            market_direction = 'negative'
        else:
            market_direction = 'strongly negative'
        
        return {
            'performance_summary': f"The Asia tech sector is showing {market_direction} performance with an average change of {avg_change:.2f}%",
            'top_performers': changes[:3] if len(changes) >= 3 else changes,
            'bottom_performers': changes[-3:] if len(changes) >= 3 else changes,
            'average_change': avg_change
        }
    
    def analyze_earnings_surprises(self, earnings_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze earnings surprises from provided data.
        
        Args:
            earnings_data: List of dictionaries with earnings data
            
        Returns:
            Dictionary with analysis results
        """
        if not earnings_data:
            return {
                'surprise_summary': 'No earnings data available for analysis',
                'positive_surprises': [],
                'negative_surprises': [],
                'average_surprise': None
            }
        
        # Filter for entries with surprise percentage
        surprises = []
        for earning in earnings_data:
            if 'surprise_pct' in earning and earning['surprise_pct'] is not None:
                surprises.append({
                    'symbol': earning['symbol'],
                    'name': earning.get('name', earning['symbol']),
                    'surprise_pct': earning['surprise_pct'],
                    'date': earning.get('date', 'Unknown')
                })
        
        # Sort by surprise percentage
        surprises.sort(key=lambda x: x['surprise_pct'], reverse=True)
        
        # Split into positive and negative surprises
        positive_surprises = [s for s in surprises if s['surprise_pct'] > 0]
        negative_surprises = [s for s in surprises if s['surprise_pct'] < 0]
        
        # Calculate average surprise
        avg_surprise = sum(item['surprise_pct'] for item in surprises) / len(surprises) if surprises else 0
        
        return {
            'surprise_summary': f"Analysis of {len(surprises)} earnings reports shows an average surprise of {avg_surprise:.2f}%",
            'positive_surprises': positive_surprises,
            'negative_surprises': negative_surprises,
            'average_surprise': avg_surprise
        }
    
    def analyze_market_sentiment(self, sentiment_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market sentiment from sentiment indicators and news.
        
        Args:
            sentiment_data: Dictionary with sentiment indicators
            news_data: List of dictionaries with news data
            
        Returns:
            Dictionary with analysis results
        """
        # Extract sentiment score and overall sentiment
        sentiment_score = sentiment_data.get('sentiment_score', 50)
        overall_sentiment = sentiment_data.get('overall_sentiment', 'neutral')
        
        # Analyze news sentiment (simple keyword-based approach)
        positive_keywords = ['growth', 'gain', 'positive', 'rise', 'up', 'beat', 'exceed', 'outperform']
        negative_keywords = ['decline', 'drop', 'negative', 'fall', 'down', 'miss', 'underperform', 'concern']
        
        news_sentiment_scores = []
        for news in news_data:
            title = news.get('title', '').lower()
            summary = news.get('summary', '').lower()
            text = title + ' ' + summary
            
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            # Calculate a simple sentiment score (-100 to 100)
            if positive_count + negative_count > 0:
                news_score = 100 * (positive_count - negative_count) / (positive_count + negative_count)
            else:
                news_score = 0
                
            news_sentiment_scores.append(news_score)
        
        # Calculate average news sentiment
        avg_news_sentiment = sum(news_sentiment_scores) / len(news_sentiment_scores) if news_sentiment_scores else 0
        
        # Combine sentiment indicators and news sentiment
        combined_score = (sentiment_score + (avg_news_sentiment + 100) / 2) / 2 if news_sentiment_scores else sentiment_score
        
        # Determine combined sentiment
        if combined_score <= 25:
            combined_sentiment = 'very negative'
        elif combined_score <= 45:
            combined_sentiment = 'negative'
        elif combined_score <= 55:
            combined_sentiment = 'neutral'
        elif combined_score <= 75:
            combined_sentiment = 'positive'
        else:
            combined_sentiment = 'very positive'
        
        return {
            'sentiment_summary': f"Market sentiment is {combined_sentiment} with a score of {combined_score:.1f}/100",
            'indicator_sentiment': overall_sentiment,
            'news_sentiment': 'positive' if avg_news_sentiment > 0 else 'negative' if avg_news_sentiment < 0 else 'neutral',
            'combined_score': combined_score,
            'key_factors': sentiment_data.get('key_indicators', [])
        }
    
    def analyze_risk_exposure(self, portfolio_data: Dict[str, Any], market_data: List[Dict[str, Any]], sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk exposure based on portfolio allocation, market data, and sentiment.
        
        Args:
            portfolio_data: Dictionary with portfolio allocation data
            market_data: List of dictionaries with market data
            sentiment_data: Dictionary with sentiment analysis
            
        Returns:
            Dictionary with risk analysis
        """
        # Extract portfolio metrics
        allocation_pct = portfolio_data.get('asia_tech_allocation_pct', 0)
        prev_allocation_pct = portfolio_data.get('previous_allocation_pct', 0)
        allocation_change = allocation_pct - prev_allocation_pct
        
        # Calculate volatility from market data
        changes = [abs(stock.get('change_pct', 0)) for stock in market_data if 'change_pct' in stock]
        volatility = sum(changes) / len(changes) if changes else 0
        
        # Get sentiment score
        sentiment_score = sentiment_data.get('combined_score', 50)
        
        # Calculate risk metrics
        # Higher allocation + high volatility + low sentiment = higher risk
        risk_score = (allocation_pct / 100) * (1 + volatility / 10) * (1 + (100 - sentiment_score) / 100)
        risk_score = min(10, risk_score * 10)  # Scale to 0-10
        
        # Determine risk level
        if risk_score < 3:
            risk_level = 'low'
        elif risk_score < 6:
            risk_level = 'moderate'
        elif risk_score < 8:
            risk_level = 'high'
        else:
            risk_level = 'very high'
        
        # Generate risk factors
        risk_factors = []
        
        if allocation_pct > 25:
            risk_factors.append(f"High portfolio concentration ({allocation_pct:.1f}%) in Asia tech sector")
        
        if allocation_change > 5:
            risk_factors.append(f"Significant increase ({allocation_change:.1f}%) in sector allocation")
        
        if volatility > 3:
            risk_factors.append(f"High market volatility ({volatility:.1f}%) in the sector")
        
        if sentiment_score < 40:
            risk_factors.append(f"Negative market sentiment (score: {sentiment_score:.1f})")
        
        # Add specific stock risks
        for stock in market_data:
            if stock.get('change_pct', 0) < -5:
                risk_factors.append(f"{stock.get('name', stock.get('symbol', ''))} showing significant decline ({stock.get('change_pct', 0):.1f}%)")
        
        return {
            'risk_summary': f"Current risk exposure to Asia tech stocks is {risk_level.upper()} with a risk score of {risk_score:.1f}/10",
            'allocation_risk': f"{allocation_pct:.1f}% of portfolio allocated to Asia tech (change: {allocation_change:+.1f}%)",
            'market_risk': f"Market volatility: {volatility:.1f}%, Sentiment score: {sentiment_score:.1f}",
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }

# Example tasks for the analysis agent
def create_analysis_tasks(agent: Agent) -> List[Task]:
    """Create tasks for the analysis agent."""
    return [
        Task(
            description="Analyze the performance of Asia tech stocks based on current market data",
            agent=agent,
            expected_output="A comprehensive analysis of stock performance including top and bottom performers and overall sector trends"
        ),
        Task(
            description="Evaluate recent earnings surprises and their impact on the Asia tech sector",
            agent=agent,
            expected_output="An analysis of earnings reports highlighting significant positive and negative surprises and their implications"
        ),
        Task(
            description="Assess the current risk exposure in the Asia tech allocation of the portfolio",
            agent=agent,
            expected_output="A detailed risk assessment including allocation risk, market risk, and specific risk factors to monitor"
        )
    ]