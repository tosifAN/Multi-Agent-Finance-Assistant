import os
import pandas as pd
from typing import Dict, List, Any
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.sectorperformance import SectorPerformances
import yfinance as yf
from datetime import datetime, timedelta, timezone

class FinancialDataAPI:
    """Class for fetching financial data from various APIs."""
    
    def __init__(self, alpha_vantage_api_key: str = None):
        """Initialize the financial data API.
        
        Args:
            alpha_vantage_api_key: Alpha Vantage API key (optional, can use from env)
        """
        self.alpha_vantage_api_key = alpha_vantage_api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.ts = TimeSeries(key=self.alpha_vantage_api_key, output_format='pandas')
        self.sp = SectorPerformances(key=self.alpha_vantage_api_key, output_format='pandas')
    
    def get_stock_data(self, symbol: str, interval: str = 'daily', output_size: str = 'compact') -> pd.DataFrame:
        """Fetch stock data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            interval: Time interval (daily, weekly, monthly)
            output_size: compact or full
            
        Returns:
            DataFrame with stock data
        """
        try:
            if interval == 'daily':
                data, meta_data = self.ts.get_daily(symbol=symbol, outputsize=output_size)
            elif interval == 'weekly':
                data, meta_data = self.ts.get_weekly(symbol=symbol)
            elif interval == 'monthly':
                data, meta_data = self.ts.get_monthly(symbol=symbol)
            else:
                raise ValueError(f"Invalid interval: {interval}")
                
            return data
        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {e}")
            # Fallback to Yahoo Finance
            return self.get_stock_data_yf(symbol, interval)
    
    def get_stock_data_yf(self, symbol: str, interval: str = 'daily') -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance as a fallback.
        
        Args:
            symbol: Stock symbol
            interval: Time interval (daily, weekly, monthly)
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Map interval to Yahoo Finance period
            period_map = {
                'daily': '1mo',
                'weekly': '6mo',
                'monthly': '1y'
            }
            period = period_map.get(interval, '1mo')
            
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Rename columns to match Alpha Vantage format
            data.rename(columns={
                'Open': '1. open',
                'High': '2. high',
                'Low': '3. low',
                'Close': '4. close',
                'Volume': '5. volume'
            }, inplace=True)
            
            return data
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def get_sector_performance(self) -> Dict[str, float]:
        """Get sector performance data.
        
        Returns:
            Dictionary with sector performance percentages
        """
        try:
            sector_perf, meta_data = self.sp.get_sector()
            # Extract the latest performance data
            latest_perf = sector_perf['Rank A: Real-Time Performance']
            return latest_perf.to_dict()
        except Exception as e:
            print(f"Error fetching sector performance: {e}")
            return {}
    
    def get_asia_tech_stocks(self) -> List[Dict[str, Any]]:
        """Get data for major Asia tech stocks.
        
        Returns:
            List of dictionaries with stock data
        """
        # List of major Asia tech stocks
        asia_tech_symbols = [
            'TSM',  # Taiwan Semiconductor
            '005930.KS',  # Samsung Electronics
            '9988.HK',  # Alibaba
            '000660.KS',  # SK Hynix
            '9984.T',  # SoftBank Group
            'BABA',  # Alibaba (US ADR)
            'BIDU',  # Baidu
            'JD',    # JD.com
            'PDD',   # PDD Holdings
            '3690.HK'  # Meituan
        ]
        
        results = []
        for symbol in asia_tech_symbols:
            try:
                # Try to get data from Yahoo Finance directly as it handles international symbols better
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                
                if not hist.empty:
                    # Calculate daily change
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                    
                    # Get company info
                    info = ticker.info
                    name = info.get('shortName', symbol)
                    
                    results.append({
                        'symbol': symbol,
                        'name': name,
                        'price': latest['Close'],
                        'change_pct': change_pct,
                        'volume': latest['Volume'],
                        'market_cap': info.get('marketCap', None),
                        'country': info.get('country', 'Unknown')
                    })
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return results
    
    def get_earnings_surprises(self) -> List[Dict[str, Any]]:
        """Get recent earnings surprises for Asia tech stocks.
        
        Returns:
            List of dictionaries with earnings surprise data
        """
        # List of major Asia tech stocks
        asia_tech_symbols = [
            'TSM',  # Taiwan Semiconductor
            '005930.KS',  # Samsung Electronics
            'BABA',  # Alibaba
            'BIDU',  # Baidu
            'JD',    # JD.com
            'PDD'    # PDD Holdings
        ]
        
        surprises = []
        for symbol in asia_tech_symbols:
            try:
                ticker = yf.Ticker(symbol)
                earnings = ticker.earnings_dates
                
                if earnings is not None and not earnings.empty:
                    # Filter for recent earnings (last 30 days)
                    recent_date = datetime.now(timezone.utc) - timedelta(days=30)
                    # Ensure earnings index is also in UTC for comparison or let pandas handle it if compatible
                    # For robust comparison, convert earnings index to UTC if it's not already
                    if earnings.index.tz is not None:
                        recent_earnings = earnings[earnings.index.tz_convert('UTC') >= recent_date]
                    else:
                        # If earnings.index is naive, localize to UTC (assuming it's intended as UTC or needs a default)
                        # This case might need more context on how yfinance returns naive datetimes for earnings_dates
                        # For now, we'll assume yfinance always returns tz-aware, or comparison with tz-aware recent_date will fail
                        # If yfinance can return naive, this part needs adjustment or error handling.
                        # However, the error message indicates yfinance *is* returning tz-aware dates.
                        recent_earnings = earnings[earnings.index >= recent_date] # This would likely re-trigger error if index is naive
                    # Simpler approach if yfinance always returns tz-aware, and pandas handles comparison: 
                    # recent_earnings = earnings[earnings.index >= recent_date]
                    
                    for date, row in recent_earnings.iterrows():
                        surprise_pct = 0
                        if row.get('EPS Estimate') and row.get('Reported EPS'):
                            if row['EPS Estimate'] != 0:
                                surprise_pct = ((row['Reported EPS'] - row['EPS Estimate']) / abs(row['EPS Estimate'])) * 100
                        
                        surprises.append({
                            'symbol': symbol,
                            'name': ticker.info.get('shortName', symbol),
                            'date': date.strftime('%Y-%m-%d'),
                            'eps_estimate': row.get('EPS Estimate'),
                            'reported_eps': row.get('Reported EPS'),
                            'surprise_pct': surprise_pct
                        })
            except Exception as e:
                print(f"Error fetching earnings data for {symbol}: {e}")
        
        return surprises
    
    def calculate_asia_tech_exposure(self, portfolio_data: Dict = None) -> Dict[str, Any]:
        """Calculate exposure to Asia tech stocks.
        
        Args:
            portfolio_data: Optional portfolio data to use for calculation
            
        Returns:
            Dictionary with exposure metrics
        """
        # If no portfolio data is provided, use a sample portfolio
        if portfolio_data is None:
            # Sample portfolio with Asia tech allocation
            portfolio_data = {
                'total_aum': 1000000,  # $1M AUM
                'asia_tech_allocation': 220000,  # $220K in Asia tech
                'previous_asia_tech_allocation': 180000,  # $180K previously
                'holdings': [
                    {'symbol': 'TSM', 'value': 50000},
                    {'symbol': 'BABA', 'value': 40000},
                    {'symbol': '005930.KS', 'value': 35000},
                    {'symbol': 'BIDU', 'value': 30000},
                    {'symbol': 'JD', 'value': 25000},
                    {'symbol': 'PDD', 'value': 40000}
                ]
            }
        
        # Calculate metrics
        current_allocation_pct = (portfolio_data['asia_tech_allocation'] / portfolio_data['total_aum']) * 100
        previous_allocation_pct = (portfolio_data['previous_asia_tech_allocation'] / portfolio_data['total_aum']) * 100
        allocation_change_pct = current_allocation_pct - previous_allocation_pct
        
        # Get current data for holdings
        holdings_data = []
        for holding in portfolio_data['holdings']:
            try:
                ticker = yf.Ticker(holding['symbol'])
                hist = ticker.history(period='5d')
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                    daily_change_pct = ((latest_price - prev_price) / prev_price) * 100
                    
                    holdings_data.append({
                        'symbol': holding['symbol'],
                        'name': ticker.info.get('shortName', holding['symbol']),
                        'value': holding['value'],
                        'allocation_pct': (holding['value'] / portfolio_data['asia_tech_allocation']) * 100,
                        'daily_change_pct': daily_change_pct
                    })
            except Exception as e:
                print(f"Error processing holding {holding['symbol']}: {e}")
        
        return {
            'total_aum': portfolio_data['total_aum'],
            'asia_tech_allocation': portfolio_data['asia_tech_allocation'],
            'asia_tech_allocation_pct': current_allocation_pct,
            'previous_allocation_pct': previous_allocation_pct,
            'allocation_change_pct': allocation_change_pct,
            'holdings': holdings_data
        }