import os
import pandas as pd
from typing import Dict, List, Optional, Union
from PyPDF2 import PdfReader
from docx import Document
import json
import csv

class DocumentLoader:
    """Base class for document loaders"""
    
    def load(self, file_path: str) -> Dict:
        """Load a document and return its content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing the document content and metadata
        """
        raise NotImplementedError("Subclasses must implement this method")


class PDFLoader(DocumentLoader):
    """Loader for PDF documents"""
    
    def load(self, file_path: str) -> Dict:
        """Load a PDF document
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing the PDF content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        
        # Extract text from each page
        text = ""
        for i in range(num_pages):
            page = reader.pages[i]
            text += page.extract_text()
        
        # Extract metadata
        metadata = {}
        if reader.metadata:
            for key, value in reader.metadata.items():
                if key.startswith('/'):
                    metadata[key[1:]] = value
                else:
                    metadata[key] = value
        
        return {
            "content": text,
            "metadata": metadata,
            "num_pages": num_pages,
            "file_path": file_path,
            "file_type": "pdf"
        }


class DocxLoader(DocumentLoader):
    """Loader for Microsoft Word documents"""
    
    def load(self, file_path: str) -> Dict:
        """Load a Word document
        
        Args:
            file_path: Path to the Word file
            
        Returns:
            Dictionary containing the Word document content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        doc = Document(file_path)
        
        # Extract text from paragraphs
        paragraphs = [p.text for p in doc.paragraphs]
        text = "\n".join(paragraphs)
        
        # Extract metadata
        metadata = {
            "core_properties": {
                "author": doc.core_properties.author,
                "created": doc.core_properties.created,
                "modified": doc.core_properties.modified,
                "title": doc.core_properties.title,
                "subject": doc.core_properties.subject
            }
        }
        
        return {
            "content": text,
            "metadata": metadata,
            "paragraphs": paragraphs,
            "file_path": file_path,
            "file_type": "docx"
        }


class TextLoader(DocumentLoader):
    """Loader for plain text documents"""
    
    def load(self, file_path: str) -> Dict:
        """Load a text document
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary containing the text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return {
            "content": text,
            "file_path": file_path,
            "file_type": "txt"
        }


class CSVLoader(DocumentLoader):
    """Loader for CSV documents"""
    
    def load(self, file_path: str) -> Dict:
        """Load a CSV document
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing the CSV content as a pandas DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to records for easier processing
        records = df.to_dict(orient='records')
        
        return {
            "dataframe": df,
            "records": records,
            "columns": df.columns.tolist(),
            "num_rows": len(df),
            "file_path": file_path,
            "file_type": "csv"
        }


class JSONLoader(DocumentLoader):
    """Loader for JSON documents"""
    
    def load(self, file_path: str) -> Dict:
        """Load a JSON document
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        return {
            "data": data,
            "file_path": file_path,
            "file_type": "json"
        }


class DocumentLoaderFactory:
    """Factory for creating document loaders based on file type"""
    
    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        """Get the appropriate loader for a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocumentLoader instance for the file type
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return PDFLoader()
        elif ext == '.docx':
            return DocxLoader()
        elif ext == '.txt':
            return TextLoader()
        elif ext == '.csv':
            return CSVLoader()
        elif ext == '.json':
            return JSONLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")


class DocumentProcessor:
    """Processor for loading and processing documents"""
    
    def __init__(self):
        """Initialize the document processor"""
        self.factory = DocumentLoaderFactory()
    
    def load_document(self, file_path: str) -> Dict:
        """Load a document
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing the document content and metadata
        """
        loader = self.factory.get_loader(file_path)
        return loader.load(file_path)
    
    def load_documents(self, file_paths: List[str]) -> List[Dict]:
        """Load multiple documents
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        documents = []
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def extract_text_from_documents(self, documents: List[Dict]) -> List[str]:
        """Extract text content from documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of text content from documents
        """
        texts = []
        for doc in documents:
            if "content" in doc:
                texts.append(doc["content"])
            elif "dataframe" in doc:
                # For CSV files, convert DataFrame to string
                texts.append(doc["dataframe"].to_string())
            elif "data" in doc:
                # For JSON files, convert data to string
                texts.append(json.dumps(doc["data"], indent=2))
        
        return texts