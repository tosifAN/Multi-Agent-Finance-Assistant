import os
import numpy as np
from typing import List, Dict, Any, Optional
from crewai import Agent, Task
# Import Pinecone and Langchain's Pinecone integration
from pinecone import Pinecone, Index
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import math

class RetrieverAgent:
    """Agent for indexing and retrieving information from a vector store."""
    
    def __init__(self, pinecone_api_key: str = None, pinecone_environment: str = None, pinecone_index_name: str = None, openai_api_key: str = None):
        """Initialize the retriever agent with Pinecone.
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment (e.g., 'gcp-starter')
            pinecone_index_name: Name of the Pinecone index
            openai_api_key: OpenAI API key for embeddings
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = pinecone_api_key or os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = pinecone_environment or os.getenv('PINECONE_ENVIRONMENT')
        self.pinecone_index_name = pinecone_index_name or os.getenv('PINECONE_INDEX_NAME')

        if not self.pinecone_api_key or not self.pinecone_environment or not self.pinecone_index_name or not self.openai_api_key:
            raise ValueError("Pinecone API key, environment, index name, and OpenAI API key must be provided or set as environment variables.")

        # Initialize Pinecone client
        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key, environment=self.pinecone_environment)

        # Check if index exists, create if not (optional, can be done manually)
        # For simplicity, we'll rely on Langchain's PineconeVectorStore to handle index existence
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Initialize the Pinecone vector store instance
        # Langchain's PineconeVectorStore handles connection and index reference
        self.vector_store = PineconeVectorStore(
            index_name=self.pinecone_index_name, 
            embedding=self.embeddings,
            pinecone_api_key=self.pinecone_api_key
            #environment=self.pinecone_environment
        )

    def create_agent(self) -> Agent:
        """Create a CrewAI agent for retrieval operations."""
        return Agent(
            role="Financial Information Retrieval Specialist",
            goal="Efficiently index and retrieve relevant financial information from Pinecone",
            backstory="""You are an expert in information retrieval systems with a 
            specialization in financial data, utilizing the power of Pinecone vector 
            database. Your expertise lies in organizing, indexing, and retrieving the 
            most relevant information from large datasets to answer specific financial 
            queries.""",
            verbose=True,
            allow_delegation=False
        )
    
    def index_documents(self, documents: List[Dict[str, str]], namespace: str = 'default') -> bool:
        """Index documents in the Pinecone vector store.
        
        Args:
            documents: List of documents to index (each with 'content' and 'metadata' keys)
            namespace: Namespace for the documents in Pinecone
            
        Returns:
            Boolean indicating success
        """
        try:
            # Convert to Document objects
            doc_objects = [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in documents]
            
            # Split documents into chunks
            splits = self.text_splitter.split_documents(doc_objects)

            print(f"this is splists ${splits}")
            print(f"this is namespace ${namespace}")
            
            # Add documents to the Pinecone index within the specified namespace
            # Langchain's add_documents handles batching and upserting


            
            self.vector_store.add_documents(splits, namespace=namespace)
            
            return True
        except Exception as e:
            print(f"Error indexing documents in Pinecone: {e}")
            return False
    
    def retrieve(self, query: str, namespace: str = 'default', k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents from the Pinecone vector store.
        
        Args:
            query: Query string
            namespace: Namespace to search in
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with content, metadata, and similarity score
        """
        try:
            # Retrieve documents from Pinecone using similarity search
            # Pinecone typically uses cosine similarity, where higher score is better (max 1.0)
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k, namespace=namespace)
            
            # Format results
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),  # Convert numpy float to Python float
                    'confidence': self._score_to_confidence(score)
                })
            
            return results
        except Exception as e:
            print(f"Error retrieving documents from Pinecone: {e}")
            return []
    
    def _score_to_confidence(self, score: float) -> float:
        """Convert similarity score (cosine similarity, 0-1) to confidence percentage.
        
        Args:
            score: Similarity score from Pinecone (typically 0 to 1 for cosine)
            
        Returns:
            Confidence percentage (0-100)
        """
        # Assuming cosine similarity where 1 is perfect match, 0 is no similarity
        # Scale the score from [0, 1] to [0, 100]
        confidence = max(0, min(100, score * 100))
        return confidence
    
    def index_financial_data(self, data: List[Dict[str, Any]], data_type: str) -> bool:
        """Index financial data in the vector store.
        
        Args:
            data: List of financial data items
            data_type: Type of data (e.g., 'news', 'earnings', 'stock_data')
            
        Returns:
            Boolean indicating success
        """
        documents = []
        
        for item in data:
            if data_type == 'news':
                # Format news articles
                content = f"Title: {item.get('title', '')}\n\nSummary: {item.get('summary', '')}\n\nSource: {item.get('source', '')}"
                metadata = {
                    'type': 'news',
                    'source': item.get('source', ''),
                    'link': item.get('link', ''),
                    'title': item.get('title', '')
                }
            elif data_type == 'earnings':
                # Format earnings data
                surprise_pct = item.get('surprise_pct', 0.0)
                if math.isnan(surprise_pct):
                    surprise_pct = 0.0  # Replace NaN with 0.0 or another default value
                content = f"Company: {item.get('name', item.get('symbol', ''))}\n\nSymbol: {item.get('symbol', '')}\n\nEPS Estimate: {item.get('eps_estimate', '')}\n\nReported EPS: {item.get('reported_eps', '')}\n\nSurprise: {surprise_pct}%"
                metadata = {
                    'type': 'earnings',
                    'symbol': item.get('symbol', ''),
                    'date': item.get('date', ''),
                    'surprise_pct': float(surprise_pct)
                }
            elif data_type == 'stock_data':
                # Format stock data
                change_pct = item.get('change_pct', 0.0)
                if math.isnan(change_pct):
                    change_pct = 0.0  # Replace NaN with 0.0 or another default value
                content = f"Company: {item.get('name', item.get('symbol', ''))}\n\nSymbol: {item.get('symbol', '')}\n\nPrice: {item.get('price', '')}\n\nChange: {change_pct}%\n\nVolume: {item.get('volume', '')}\n\nMarket Cap: {item.get('market_cap', '')}"
                metadata = {
                    'type': 'stock_data',
                    'symbol': item.get('symbol', ''),
                    'country': item.get('country', ''),
                    'change_pct': float(change_pct)
                }
            elif data_type == 'sentiment':
                # Format sentiment data
                content = f"Overall Sentiment: {item.get('overall_sentiment', '')}\n\nSentiment Score: {item.get('sentiment_score', '')}\n\nKey Indicators: {', '.join([ind.get('headline', '') for ind in item.get('key_indicators', [])])}"
                metadata = {
                    'type': 'sentiment',
                    'sentiment': item.get('overall_sentiment', ''),
                    'score': item.get('sentiment_score', None)
                }
            else:
                # Generic format for other data types
                content = str(item)
                metadata = {'type': data_type}
            
            documents.append({
                'content': content,
                'metadata': metadata
            })
        
        return self.index_documents(documents, namespace=data_type)
    
    def retrieve_asia_tech_info(self, query: str, confidence_threshold: float = 60.0) -> Dict[str, Any]:
        """Retrieve information about Asia tech stocks.
        
        Args:
            query: Query string
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with retrieved information and confidence scores
        """
        # Namespaces to search in
        namespaces = ['news', 'earnings', 'stock_data', 'sentiment', 'portfolio', 'finance']
        
        all_results = []
        confidence_levels = []
        
        for namespace in namespaces:
            results = self.retrieve(query, namespace=namespace, k=3)
            all_results.extend(results)
            
            # Track confidence levels
            for result in results:
                confidence_levels.append(result['confidence'])
        
        # Sort by confidence
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
        
        # Filter by confidence threshold
        filtered_results = [r for r in all_results if r['confidence'] >= confidence_threshold]
        
        return {
            'results': filtered_results,
            'avg_confidence': avg_confidence,
            'below_threshold': avg_confidence < confidence_threshold,
            'top_result': filtered_results[0] if filtered_results else None
        }

# Example tasks for the retriever agent
def create_retriever_tasks(agent: Agent) -> List[Task]:
    """Create tasks for the retriever agent."""
    return [
        Task(
            description="Index recent financial news, earnings reports, and stock data for Asia tech companies",
            agent=agent,
            expected_output="Confirmation that all data has been successfully indexed in the vector store"
        ),
        Task(
            description="Retrieve relevant information about risk exposure in Asia tech stocks",
            agent=agent,
            expected_output="A collection of the most relevant documents about current risk factors and exposure levels in Asia tech stocks"
        ),
        Task(
            description="Find information about recent earnings surprises in the Asia tech sector",
            agent=agent,
            expected_output="Detailed information about companies that have reported earnings significantly different from analyst expectations"
        )
    ]