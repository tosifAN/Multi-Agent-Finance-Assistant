import os
from typing import List, Dict, Any
from crewai import Agent, Task
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class LanguageAgent:
    """Agent for synthesizing narratives from financial data."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the language agent.
        
        Args:
            openai_api_key: OpenAI API key (optional, can use from env)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=self.openai_api_key)
        
    def create_agent(self) -> Agent:
        """Create a CrewAI agent for language operations."""
        return Agent(
            role="Financial Narrative Specialist",
            goal="Synthesize clear, concise, and insightful financial narratives",
            backstory="""You are an expert financial writer with years of experience in 
            distilling complex financial data into clear, actionable narratives. Your 
            specialty is in creating concise market briefs that highlight the most 
            important information for busy portfolio managers.""",
            verbose=True,
            allow_delegation=False
        )
    
    def create_market_brief(self, 
                           portfolio_data: Dict[str, Any], 
                           stock_performance: Dict[str, Any], 
                           earnings_analysis: Dict[str, Any], 
                           sentiment_analysis: Dict[str, Any], 
                           risk_analysis: Dict[str, Any]) -> str:
        """Create a comprehensive market brief based on financial data.
        
        Args:
            portfolio_data: Portfolio allocation data
            stock_performance: Stock performance analysis
            earnings_analysis: Earnings surprises analysis
            sentiment_analysis: Market sentiment analysis
            risk_analysis: Risk exposure analysis
            
        Returns:
            Formatted market brief text
        """
        # Create prompt template
        template = """
        You are a financial advisor creating a morning market brief for a portfolio manager.
        Focus on Asia tech stocks exposure and earnings surprises.
        
        PORTFOLIO DATA:
        {portfolio_data}
        
        STOCK PERFORMANCE:
        {stock_performance}
        
        EARNINGS ANALYSIS:
        {earnings_analysis}
        
        SENTIMENT ANALYSIS:
        {sentiment_analysis}
        
        RISK ANALYSIS:
        {risk_analysis}
        
        Create a concise, informative market brief that highlights:
        1. Current Asia tech allocation and change from previous day
        2. Notable earnings surprises (both positive and negative)
        3. Overall market sentiment and key factors
        4. Current risk exposure and recommendations
        
        The brief should be conversational but professional, about 3-4 paragraphs long.
        """
        
        # Format the input data
        portfolio_str = f"Asia tech allocation: {portfolio_data.get('asia_tech_allocation_pct', 0):.1f}% of AUM, "
        portfolio_str += f"previous allocation: {portfolio_data.get('previous_allocation_pct', 0):.1f}%, "
        portfolio_str += f"change: {portfolio_data.get('allocation_change_pct', 0):+.1f}%"
        
        performance_str = stock_performance.get('performance_summary', 'No performance data available')
        if stock_performance.get('top_performers'):
            top = stock_performance['top_performers'][0]
            performance_str += f"\nTop performer: {top.get('name', top.get('symbol', ''))} ({top.get('change_pct', 0):+.1f}%)"
        if stock_performance.get('bottom_performers'):
            bottom = stock_performance['bottom_performers'][-1]
            performance_str += f"\nBottom performer: {bottom.get('name', bottom.get('symbol', ''))} ({bottom.get('change_pct', 0):+.1f}%)"
        
        earnings_str = earnings_analysis.get('surprise_summary', 'No earnings data available')
        for surprise in earnings_analysis.get('positive_surprises', [])[:2]:
            earnings_str += f"\n{surprise.get('name', surprise.get('symbol', ''))} beat estimates by {surprise.get('surprise_pct', 0):.1f}%"
        for surprise in earnings_analysis.get('negative_surprises', [])[:2]:
            earnings_str += f"\n{surprise.get('name', surprise.get('symbol', ''))} missed estimates by {abs(surprise.get('surprise_pct', 0)):.1f}%"
        
        sentiment_str = sentiment_analysis.get('sentiment_summary', 'No sentiment data available')
        for factor in sentiment_analysis.get('key_factors', [])[:3]:
            sentiment_str += f"\n- {factor.get('headline', '')}"
        
        risk_str = risk_analysis.get('risk_summary', 'No risk analysis available')
        risk_str += f"\n{risk_analysis.get('allocation_risk', '')}"
        for factor in risk_analysis.get('risk_factors', [])[:3]:
            risk_str += f"\n- {factor}"
        
        # Create and run the chain
        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        result = chain.run({
            'portfolio_data': portfolio_str,
            'stock_performance': performance_str,
            'earnings_analysis': earnings_str,
            'sentiment_analysis': sentiment_str,
            'risk_analysis': risk_str
        })
        
        return result.strip()
    
    def answer_specific_query(self, query: str, retrieved_data: List[Dict[str, Any]]) -> str:
        """Answer a specific query using retrieved data.
        
        Args:
            query: User query
            retrieved_data: List of retrieved documents
            
        Returns:
            Answer to the query
        """
        # Create prompt template
        template = """
        You are a financial advisor answering a question about Asia tech stocks.
        
        USER QUERY:
        {query}
        
        RETRIEVED INFORMATION:
        {retrieved_info}
        
        Provide a clear, concise answer to the query based on the retrieved information.
        If the information is insufficient to answer the query, acknowledge the limitations
        and provide the best possible answer with the available data.
        
        Your answer should be conversational but professional, about 1-2 paragraphs long.
        """
        
        # Format retrieved information
        retrieved_info = ""
        for i, doc in enumerate(retrieved_data):
            retrieved_info += f"Document {i+1}:\n"
            retrieved_info += f"Content: {doc.get('content', '')}\n"
            retrieved_info += f"Confidence: {doc.get('confidence', 0):.1f}%\n\n"
        
        if not retrieved_info:
            retrieved_info = "No relevant information found."
        
        # Create and run the chain
        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        result = chain.run({
            'query': query,
            'retrieved_info': retrieved_info
        })
        
        return result.strip()
    
    def generate_recommendations(self, 
                               portfolio_data: Dict[str, Any], 
                               stock_performance: Dict[str, Any], 
                               risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate investment recommendations based on analysis.
        
        Args:
            portfolio_data: Portfolio allocation data
            stock_performance: Stock performance analysis
            risk_analysis: Risk exposure analysis
            
        Returns:
            List of recommendation objects
        """
        # Create prompt template
        template = """
        You are a financial advisor generating investment recommendations for a portfolio manager.
        Focus on Asia tech stocks exposure and risk management.
        
        PORTFOLIO DATA:
        {portfolio_data}
        
        STOCK PERFORMANCE:
        {stock_performance}
        
        RISK ANALYSIS:
        {risk_analysis}
        
        Generate 3 specific, actionable investment recommendations based on the data.
        Each recommendation should include:
        1. A clear action (buy, sell, hold, rebalance, etc.)
        2. Specific assets or sectors involved
        3. A brief rationale for the recommendation
        
        Format each recommendation as a JSON object with 'action', 'target', and 'rationale' fields.
        """
        
        # Format the input data (similar to create_market_brief)
        portfolio_str = f"Asia tech allocation: {portfolio_data.get('asia_tech_allocation_pct', 0):.1f}% of AUM, "
        portfolio_str += f"previous allocation: {portfolio_data.get('previous_allocation_pct', 0):.1f}%, "
        portfolio_str += f"change: {portfolio_data.get('allocation_change_pct', 0):+.1f}%"
        
        performance_str = stock_performance.get('performance_summary', 'No performance data available')
        for performer in stock_performance.get('top_performers', [])[:3]:
            performance_str += f"\n{performer.get('name', performer.get('symbol', ''))} ({performer.get('change_pct', 0):+.1f}%)"
        for performer in stock_performance.get('bottom_performers', [])[-3:]:
            performance_str += f"\n{performer.get('name', performer.get('symbol', ''))} ({performer.get('change_pct', 0):+.1f}%)"
        
        risk_str = risk_analysis.get('risk_summary', 'No risk analysis available')
        risk_str += f"\n{risk_analysis.get('allocation_risk', '')}"
        for factor in risk_analysis.get('risk_factors', []):
            risk_str += f"\n- {factor}"
        
        # Create and run the chain
        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        result = chain.run({
            'portfolio_data': portfolio_str,
            'stock_performance': performance_str,
            'risk_analysis': risk_str
        })
        
        # Parse the recommendations (simplified parsing)
        recommendations = []
        try:
            # Split by recommendation
            import re
            import json
            
            # Try to extract JSON objects
            json_pattern = r'\{[^\{\}]*\"action\"[^\{\}]*\"target\"[^\{\}]*\"rationale\"[^\{\}]*\}'
            matches = re.findall(json_pattern, result)
            
            for match in matches:
                try:
                    rec = json.loads(match)
                    if 'action' in rec and 'target' in rec and 'rationale' in rec:
                        recommendations.append(rec)
                except:
                    pass
            
            # If no valid JSON found, create structured recommendations from text
            if not recommendations:
                lines = result.split('\n')
                current_rec = {}
                
                for line in lines:
                    if line.lower().startswith('recommendation'):
                        if current_rec and 'action' in current_rec:
                            recommendations.append(current_rec)
                        current_rec = {}
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == 'action':
                            current_rec['action'] = value
                        elif key in ['target', 'asset', 'sector']:
                            current_rec['target'] = value
                        elif key in ['rationale', 'reason']:
                            current_rec['rationale'] = value
                
                if current_rec and 'action' in current_rec:
                    recommendations.append(current_rec)
        except Exception as e:
            print(f"Error parsing recommendations: {e}")
            # Fallback to simple recommendations
            recommendations = [
                {
                    'action': 'Review',
                    'target': 'Asia tech allocation',
                    'rationale': 'Based on current market conditions and risk analysis'
                }
            ]
        
        return recommendations

# Example tasks for the language agent
def create_language_tasks(agent: Agent) -> List[Task]:
    """Create tasks for the language agent."""
    return [
        Task(
            description="Create a comprehensive morning market brief focusing on Asia tech stocks",
            agent=agent,
            expected_output="A concise, informative market brief highlighting current allocation, earnings surprises, sentiment, and risk exposure"
        ),
        Task(
            description="Generate specific investment recommendations based on the current analysis",
            agent=agent,
            expected_output="A list of actionable investment recommendations with clear rationales based on the data analysis"
        ),
        Task(
            description="Answer the portfolio manager's specific query about risk exposure in Asia tech stocks",
            agent=agent,
            expected_output="A clear, concise answer to the query based on the retrieved information and analysis"
        )
    ]