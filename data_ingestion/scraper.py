import os
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import pandas as pd

class FinancialScraper:
    """Class for scraping financial news and filings from web sources."""
    
    def __init__(self):
        """Initialize the financial scraper."""
        self.headers = {'User-Agent': 'Mozilla/5.0'}
    
    def scrape_financial_news(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Scrape financial news related to Asia tech stocks.
        
        Args:
            keywords: List of keywords to filter news by
            
        Returns:
            List of dictionaries with news data
        """
        if keywords is None:
            keywords = ['Asia tech', 'semiconductor', 'TSMC', 'Samsung', 'Alibaba']
        
        news_sources = [
            {
                'name': 'Yahoo Finance',
                'url': 'https://finance.yahoo.com/news/',
                'article_selector': 'li.js-stream-content',
                'title_selector': 'h3',
                'link_selector': 'a',
                'summary_selector': 'p'
            },
            {
                'name': 'CNBC Asia',
                'url': 'https://www.cnbc.com/asia-news/',
                'article_selector': '.Card-standardBreakerCard',
                'title_selector': '.Card-title',
                'link_selector': 'a',
                'summary_selector': '.Card-description'
            }
        ]
        
        all_news = []
        
        for source in news_sources:
            try:
                response = requests.get(source['url'], headers=self.headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = soup.select(source['article_selector'])
                    
                    for article in articles[:10]:  # Limit to 10 articles per source
                        try:
                            title_elem = article.select_one(source['title_selector'])
                            link_elem = article.select_one(source['link_selector'])
                            summary_elem = article.select_one(source['summary_selector'])
                            
                            if title_elem and link_elem:
                                title = title_elem.text.strip()
                                link = link_elem.get('href')
                                if not link.startswith('http'):
                                    # Handle relative URLs
                                    if link.startswith('/'):
                                        base_url = '/'.join(source['url'].split('/')[:3])
                                        link = base_url + link
                                
                                summary = summary_elem.text.strip() if summary_elem else ''
                                
                                # Check if article matches any keywords
                                if any(keyword.lower() in title.lower() or keyword.lower() in summary.lower() for keyword in keywords):
                                    all_news.append({
                                        'source': source['name'],
                                        'title': title,
                                        'link': link,
                                        'summary': summary
                                    })
                        except Exception as e:
                            print(f"Error parsing article from {source['name']}: {e}")
            except Exception as e:
                print(f"Error scraping {source['name']}: {e}")
        
        return all_news
    
    def scrape_earnings_reports(self, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """Scrape recent earnings reports for specified symbols.
        
        Args:
            symbols: List of stock symbols to get earnings for
            
        Returns:
            List of dictionaries with earnings data
        """
        if symbols is None:
            symbols = ['TSM', 'BABA', '005930.KS', 'BIDU', 'JD', 'PDD']
        
        earnings_data = []
        
        # For demonstration, we'll use a simplified approach that scrapes Yahoo Finance earnings pages
        for symbol in symbols:
            try:
                url = f"https://finance.yahoo.com/quote/{symbol}/analysis"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract earnings data from the page
                    earnings_tables = soup.find_all('table')
                    for table in earnings_tables:
                        table_title = table.find_previous('h2')
                        if table_title and 'Earnings Estimate' in table_title.text:
                            rows = table.find_all('tr')
                            headers = [th.text.strip() for th in rows[0].find_all('th')]
                            
                            for row in rows[1:]:
                                cells = row.find_all('td')
                                if len(cells) >= len(headers):
                                    row_data = {headers[i]: cells[i].text.strip() for i in range(len(headers))}
                                    row_data['symbol'] = symbol
                                    earnings_data.append(row_data)
            except Exception as e:
                print(f"Error scraping earnings for {symbol}: {e}")
        
        return earnings_data
    
    def scrape_market_sentiment(self) -> Dict[str, Any]:
        """Scrape market sentiment indicators for Asia tech sector.
        
        Returns:
            Dictionary with sentiment data
        """
        sentiment_sources = [
            {
                'name': 'CNN Fear & Greed Index',
                'url': 'https://www.cnn.com/markets/fear-and-greed',
                'indicator_selector': '.market-fng-gauge__dial-number'
            },
            {
                'name': 'MarketWatch Asia Markets',
                'url': 'https://www.marketwatch.com/markets/asia',
                'indicator_selector': '.element--article'
            }
        ]
        
        sentiment_data = {
            'sources': [],
            'overall_sentiment': 'neutral',  # Default sentiment
            'sentiment_score': 50,  # Default score (0-100, where 0 is extreme fear, 100 is extreme greed)
            'key_indicators': []
        }
        
        for source in sentiment_sources:
            try:
                response = requests.get(source['url'], headers=self.headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    if source['name'] == 'CNN Fear & Greed Index':
                        indicator = soup.select_one(source['indicator_selector'])
                        if indicator:
                            try:
                                score = int(indicator.text.strip())
                                sentiment_data['sentiment_score'] = score
                                
                                # Determine sentiment based on score
                                if score <= 25:
                                    sentiment = 'extreme fear'
                                elif score <= 45:
                                    sentiment = 'fear'
                                elif score <= 55:
                                    sentiment = 'neutral'
                                elif score <= 75:
                                    sentiment = 'greed'
                                else:
                                    sentiment = 'extreme greed'
                                    
                                sentiment_data['overall_sentiment'] = sentiment
                                sentiment_data['sources'].append({
                                    'name': source['name'],
                                    'score': score,
                                    'sentiment': sentiment
                                })
                            except ValueError:
                                pass
                    
                    elif source['name'] == 'MarketWatch Asia Markets':
                        articles = soup.select(source['indicator_selector'])[:5]  # Get top 5 articles
                        for article in articles:
                            title_elem = article.select_one('h3')
                            if title_elem:
                                title = title_elem.text.strip()
                                sentiment_data['key_indicators'].append({
                                    'source': 'MarketWatch',
                                    'headline': title
                                })
            except Exception as e:
                print(f"Error scraping sentiment from {source['name']}: {e}")
        
        return sentiment_data