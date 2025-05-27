import os
import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union

# Load environment variables
load_dotenv()

class AlphaVantageClient:
    """Client for fetching data from Alpha Vantage API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Alpha Vantage client with API key"""
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_stock_data(self, symbol: str, function: str = "TIME_SERIES_DAILY", outputsize: str = "compact") -> Dict:
        """Fetch stock data from Alpha Vantage
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT)
            function: Alpha Vantage function (TIME_SERIES_DAILY, TIME_SERIES_WEEKLY, etc.)
            outputsize: compact (100 data points) or full (20+ years of data)
            
        Returns:
            Dictionary containing the stock data
        """
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_sector_performance(self) -> Dict:
        """Fetch sector performance data
        
        Returns:
            Dictionary containing sector performance data
        """
        params = {
            "function": "SECTOR",
            "apikey": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_earnings_calendar(self, symbol: Optional[str] = None, horizon: str = "3month") -> Dict:
        """Fetch earnings calendar data
        
        Args:
            symbol: Optional stock symbol to filter results
            horizon: Time horizon (3month, 6month, 12month)
            
        Returns:
            Dictionary containing earnings calendar data
        """
        params = {
            "function": "EARNINGS_CALENDAR",
            "horizon": horizon,
            "apikey": self.api_key
        }
        
        if symbol:
            params["symbol"] = symbol
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()


class YahooFinanceClient:
    """Client for fetching data from Yahoo Finance"""
    
    def get_stock_data(self, symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame containing the stock data
        """
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period, interval=interval)
    
    def get_multiple_stocks_data(self, symbols: List[str], period: str = "1mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_stock_data(symbol, period, interval)
        return data
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Fetch detailed information about a stock
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing stock information
        """
        ticker = yf.Ticker(symbol)
        return ticker.info
    
    def get_asia_tech_stocks(self) -> List[str]:
        """Get a list of major Asia tech stocks
        
        Returns:
            List of stock symbols for major Asian tech companies
        """
        # This is a simplified list - in a real implementation, this could be more comprehensive
        # or fetched from an external source
        return [
            "9988.HK",  # Alibaba (Hong Kong)
            "0700.HK",  # Tencent (Hong Kong)
            "2330.TW",  # TSMC (Taiwan)
            "005930.KS", # Samsung (South Korea)
            "6758.T",    # Sony (Japan)
            "4689.T",    # Yahoo Japan
            "9984.T",    # SoftBank (Japan)
            "6501.T",    # Hitachi (Japan)
            "BABA",      # Alibaba (US ADR)
            "BIDU"       # Baidu (US ADR)
        ]


class MarketDataClient:
    """Unified client for fetching market data from multiple sources"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """Initialize market data client with API keys"""
        self.alpha_vantage = AlphaVantageClient(alpha_vantage_key)
        self.yahoo_finance = YahooFinanceClient()
    
    def get_asia_tech_exposure(self) -> Dict:
        """Calculate exposure to Asia tech stocks
        
        Returns:
            Dictionary containing exposure metrics
        """
        # Get list of Asia tech stocks
        asia_tech_symbols = self.yahoo_finance.get_asia_tech_stocks()
        
        # Get market data for these stocks
        stock_data = self.yahoo_finance.get_multiple_stocks_data(asia_tech_symbols, period="5d")
        
        # Calculate market cap and other metrics
        market_caps = {}
        daily_changes = {}
        
        for symbol in asia_tech_symbols:
            try:
                # Get stock info for market cap
                info = self.yahoo_finance.get_stock_info(symbol)
                market_caps[symbol] = info.get("marketCap", 0)
                
                # Calculate daily change
                if symbol in stock_data and not stock_data[symbol].empty:
                    df = stock_data[symbol]
                    if len(df) >= 2:
                        yesterday = df.iloc[-2]
                        today = df.iloc[-1]
                        daily_changes[symbol] = (today["Close"] - yesterday["Close"]) / yesterday["Close"] * 100
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        # Calculate total market cap and weighted average change
        total_market_cap = sum(market_caps.values())
        weighted_change = sum(daily_changes.get(s, 0) * market_caps.get(s, 0) for s in asia_tech_symbols) / total_market_cap if total_market_cap else 0
        
        return {
            "total_market_cap": total_market_cap,
            "daily_weighted_change": weighted_change,
            "stock_data": stock_data,
            "market_caps": market_caps,
            "daily_changes": daily_changes
        }
    
    def get_earnings_surprises(self, days_back: int = 7) -> List[Dict]:
        """Get recent earnings surprises
        
        Args:
            days_back: Number of days to look back for earnings reports
            
        Returns:
            List of dictionaries containing earnings surprise data
        """
        # This would typically use Alpha Vantage's earnings calendar API
        # For demonstration, we'll return some sample data
        # In a real implementation, this would fetch and process actual earnings data
        
        # Sample data - in a real implementation, this would be fetched from the API
        return [
            {
                "symbol": "2330.TW",  # TSMC
                "name": "Taiwan Semiconductor Manufacturing Company",
                "report_date": "2023-04-20",
                "estimate": 1.50,
                "actual": 1.56,
                "surprise_percent": 4.0
            },
            {
                "symbol": "005930.KS",  # Samsung
                "name": "Samsung Electronics Co Ltd",
                "report_date": "2023-04-27",
                "estimate": 1.25,
                "actual": 1.225,
                "surprise_percent": -2.0
            }
        ]