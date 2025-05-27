import os
import json
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
import numpy as np

# Import necessary modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_ingestion.document_loader import DocumentProcessor

# Define request and response models
class RetrieverRequest(BaseModel):
    query: str
    collection: str  # 'market_news', 'earnings_reports', 'financial_data'
    top_k: int = 5
    parameters: Optional[Dict[str, Any]] = None

class RetrieverResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str

# Create the FastAPI app
app = FastAPI(title="Retriever Agent", description="Agent for retrieving relevant information from vector stores")

# Vector store interface
class VectorStore:
    """Base class for vector stores"""
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """Add texts to the vector store
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of IDs for the added texts
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata
        """
        raise NotImplementedError("Subclasses must implement this method")


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize FAISS vector store
        
        Args:
            index_path: Optional path to load an existing index
        """
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install faiss-cpu and sentence-transformers: pip install faiss-cpu sentence-transformers")
        
        self.faiss = faiss
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadatas = []
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """Add texts to the FAISS index
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of IDs for the added texts
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Convert texts to embeddings
        embeddings = self.model.encode(texts)
        
        # Add to index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store texts and metadata
        start_idx = len(self.texts)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        # Return IDs
        return [str(i) for i in range(start_idx, len(self.texts))]
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts in the FAISS index
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata
        """
        # Convert query to embedding
        query_embedding = self.model.encode([query])
        
        # Search index
        k = min(k, len(self.texts))
        if k == 0:
            return []
        
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(distances[0][i])
                })
        
        return results
    
    def save_index(self, index_path: str):
        """Save the FAISS index to disk
        
        Args:
            index_path: Path to save the index
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save the index
        self.faiss.write_index(self.index, index_path)
        
        # Save texts and metadata
        with open(f"{index_path}.json", 'w') as f:
            json.dump({"texts": self.texts, "metadatas": self.metadatas}, f)
    
    def load_index(self, index_path: str):
        """Load a FAISS index from disk
        
        Args:
            index_path: Path to the index file
        """
        # Load the index
        self.index = self.faiss.read_index(index_path)
        
        # Load texts and metadata
        if os.path.exists(f"{index_path}.json"):
            with open(f"{index_path}.json", 'r') as f:
                data = json.load(f)
                self.texts = data["texts"]
                self.metadatas = data["metadatas"]


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation"""
    
    def __init__(self, index_name: str = "finance-assistant"):
        """Initialize Pinecone vector store
        
        Args:
            index_name: Name of the Pinecone index
        """
        try:
            import pinecone
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install pinecone-client and sentence-transformers: pip install pinecone-client sentence-transformers")
        
        self.pinecone = pinecone
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Pinecone API key is required")
        
        self.pinecone.init(api_key=api_key, environment="us-west1-gcp")
        
        # Create index if it doesn't exist
        if index_name not in self.pinecone.list_indexes():
            self.pinecone.create_index(name=index_name, dimension=self.dimension)
        
        self.index = self.pinecone.Index(index_name)
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> List[str]:
        """Add texts to the Pinecone index
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of IDs for the added texts
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Convert texts to embeddings
        embeddings = self.model.encode(texts)
        
        # Prepare vectors for Pinecone
        ids = [str(i) for i in range(len(texts))]
        vectors = []
        
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            # Add text to metadata
            metadata["text"] = text
            
            vectors.append({
                "id": ids[i],
                "values": embedding.tolist(),
                "metadata": metadata
            })
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        
        return ids
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar texts in the Pinecone index
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing text and metadata
        """
        # Convert query to embedding
        query_embedding = self.model.encode(query)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True
        )
        
        # Format results
        formatted_results = []
        for match in results["matches"]:
            metadata = match["metadata"]
            text = metadata.pop("text", "")
            
            formatted_results.append({
                "text": text,
                "metadata": metadata,
                "score": match["score"]
            })
        
        return formatted_results


class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def get_vector_store(store_type: str = "faiss", **kwargs) -> VectorStore:
        """Get a vector store instance
        
        Args:
            store_type: Type of vector store ('faiss' or 'pinecone')
            **kwargs: Additional arguments for the vector store
            
        Returns:
            VectorStore instance
        """
        if store_type.lower() == "faiss":
            return FAISSVectorStore(**kwargs)
        elif store_type.lower() == "pinecone":
            return PineconeVectorStore(**kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")


class RetrieverAgent:
    """Agent for retrieving relevant information from vector stores"""
    
    def __init__(self, vector_store_type: str = "faiss"):
        """Initialize the retriever agent
        
        Args:
            vector_store_type: Type of vector store to use ('faiss' or 'pinecone')
        """
        self.vector_store_type = vector_store_type
        self.vector_stores = {}
        self.document_processor = DocumentProcessor()
    
    def get_vector_store(self, collection: str) -> VectorStore:
        """Get or create a vector store for a collection
        
        Args:
            collection: Name of the collection
            
        Returns:
            VectorStore instance
        """
        if collection not in self.vector_stores:
            # Create vector store
            if self.vector_store_type == "faiss":
                index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
                collection_path = f"{index_path}/{collection}"
                self.vector_stores[collection] = VectorStoreFactory.get_vector_store(
                    store_type="faiss",
                    index_path=collection_path if os.path.exists(collection_path) else None
                )
            else:  # pinecone
                self.vector_stores[collection] = VectorStoreFactory.get_vector_store(
                    store_type="pinecone",
                    index_name=f"finance-assistant-{collection}"
                )
        
        return self.vector_stores[collection]
    
    async def index_documents(self, collection: str, documents: List[Dict]) -> Dict:
        """Index documents in a vector store
        
        Args:
            collection: Name of the collection
            documents: List of document dictionaries
            
        Returns:
            Dictionary containing indexing results
        """
        # Extract text from documents
        texts = self.document_processor.extract_text_from_documents(documents)
        
        # Get vector store
        vector_store = self.get_vector_store(collection)
        
        # Add texts to vector store
        ids = vector_store.add_texts(texts, metadatas=[doc.get("metadata", {}) for doc in documents])
        
        # Save index if using FAISS
        if self.vector_store_type == "faiss":
            index_path = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
            collection_path = f"{index_path}/{collection}"
            os.makedirs(os.path.dirname(collection_path), exist_ok=True)
            vector_store.save_index(collection_path)
        
        return {
            "indexed_count": len(ids),
            "collection": collection
        }
    
    async def retrieve(self, query: str, collection: str, top_k: int = 5) -> Dict:
        """Retrieve relevant documents for a query
        
        Args:
            query: Query string
            collection: Name of the collection to search
            top_k: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # Get vector store
        vector_store = self.get_vector_store(collection)
        
        # Search for similar documents
        results = vector_store.similarity_search(query, k=top_k)
        
        # Calculate confidence score (normalized)
        confidence = 0.0
        if results:
            # Convert distance to similarity score (1 - normalized distance)
            scores = [result["score"] for result in results]
            max_score = max(scores) if scores else 0
            min_score = min(scores) if scores else 0
            score_range = max_score - min_score if max_score > min_score else 1
            
            # Normalize scores to 0-1 range and average them
            normalized_scores = [(score - min_score) / score_range for score in scores]
            confidence = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0
        
        return {
            "results": results,
            "count": len(results),
            "confidence": confidence,
            "collection": collection
        }
    
    async def process_request(self, request: RetrieverRequest) -> Dict:
        """Process a retriever request
        
        Args:
            request: RetrieverRequest object
            
        Returns:
            Dictionary containing the response data
        """
        try:
            # Retrieve documents
            results = await self.retrieve(
                query=request.query,
                collection=request.collection,
                top_k=request.top_k
            )
            
            return {
                "status": "success",
                "data": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }


# Dependency to get the retriever agent
def get_retriever_agent():
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss")
    return RetrieverAgent(vector_store_type=vector_store_type)


# API routes
@app.post("/api/retrieve", response_model=RetrieverResponse)
async def retrieve_data(request: RetrieverRequest, 
                       retriever_agent: RetrieverAgent = Depends(get_retriever_agent)):
    """Endpoint for retrieving relevant information"""
    response = await retriever_agent.process_request(request)
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Run the FastAPI app if this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)