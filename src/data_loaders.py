# Import required libraries
import pandas as pd
from pathlib import Path
from typing import List
from langchain.schema import Document
import logging

# For PDF processing - now using LangChain's PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

# Configure logging
logger = logging.getLogger(__name__)

# --- Existing functions (as provided by user) ---

# Define a placeholder for COMPANY_INFO_DIR if it's not defined in config.py
# In a real application, ensure config.py is accessible or pass this path.
try:
    from config import COMPANY_INFO_DIR
except ImportError:
    logger.warning("COMPANY_INFO_DIR not found in config.py. Using a default placeholder.")
    COMPANY_INFO_DIR = Path("./company_info") # Placeholder path, adjust as needed

def load_faq_documents(faq_path: Path = Path(COMPANY_INFO_DIR) / "FAQ.csv") -> List[Document]:
    """
    Load and process FAQ documents from CSV file.
    
    Args:
        faq_path: Path to the FAQ CSV file
        
    Returns:
        List of Document objects
    """
    try:
        # Validate file exists
        if not faq_path.exists():
            raise FileNotFoundError(f"FAQ file not found at {faq_path}")
            
        df = pd.read_csv(faq_path)
        
        # Validate required columns
        required_cols = ['Question', 'Answer']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
            
        documents = []
        for idx, row in df.iterrows():
            content = f"Question: {row.get('Question', '')}\nAnswer: {row.get('Answer', '')}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "company_faq",
                    "type": "faq", 
                    "doc_id": f"{idx}", 
                    "filename": faq_path.name
                }
            )
            documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} FAQ documents from {faq_path.name}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading FAQ documents from {faq_path.name}: {str(e)}")
        raise


def load_company_info(info_path: Path = Path(COMPANY_INFO_DIR) / "info.md") -> Document:
    """
    Load company information from markdown file.
    
    Args:
        info_path: Path to the company info markdown file
        
    Returns:
        Document object containing company info
    """
    try:
        # Validate file exists
        if not info_path.exists():
            raise FileNotFoundError(f"Info file not found at {info_path}")
            
        with open(info_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        doc = Document(
            page_content=content,
            metadata={
                "source": "company_info",
                "type": "general_info",
                "filename": info_path.name,
                "doc_id": "company_info_main"
            }
        )
        logger.info(f"Loaded company info document from {info_path.name}")
        return doc
        
    except Exception as e:
        logger.error(f"Error loading company info from {info_path.name}: {str(e)}")
        raise

# --- New functions for PDF, TXT, and Image loading ---

def load_pdf_document(file_path: Path) -> List[Document]:
    """
    Load text from a PDF file using LangChain's PyPDFLoader.
    Each page is treated as a separate document.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        A list of Document objects, one for each page of the PDF.
    """
    documents = []
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found at {file_path}")
        
        loader = PyPDFLoader(str(file_path)) # PyPDFLoader expects a string path
        docs = loader.load() # This returns a list of LangChain Document objects

        # Enhance metadata for consistency and add source/type
        for doc in docs:
            doc.metadata["source"] = "uploaded_file"
            doc.metadata["type"] = "pdf"
            doc.metadata["filename"] = file_path.name
            # PyPDFLoader usually adds 'page' and 'source' (which is the file path)
            # We can use the existing 'page' if it's there or default to 0
            page_num = doc.metadata.get("page", 0) 
            doc.metadata["doc_id"] = f"{file_path.stem}_page_{page_num + 1}" # Ensure page number is 1-indexed

        documents.extend(docs)
        
        logger.info(f"Loaded {len(documents)} pages from PDF using PyPDFLoader: {file_path.name}")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF file {file_path.name} with PyPDFLoader: {str(e)}")
        raise

def load_txt_document(file_path: Path) -> Document:
    """
    Load text from a TXT file.
    
    Args:
        file_path: Path to the TXT file.
        
    Returns:
        A Document object containing the text from the file.
    """
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"TXT file not found at {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        doc = Document(
            page_content=content,
            metadata={
                "source": "uploaded_file",
                "type": "txt",
                "filename": file_path.name,
                "doc_id": file_path.stem
            }
        )
        logger.info(f"Loaded TXT file: {file_path.name}")
        return doc
    except Exception as e:
        logger.error(f"Error loading TXT file {file_path.name}: {str(e)}")
        raise


def process_uploaded_file(file_path: Path) -> List[Document]:
    """
    Determines the file extension and calls the appropriate function to process it.
    
    Args:
        file_path: Path to the uploaded file.
        
    Returns:
        A list of Document objects containing the extracted text.
        Returns an empty list if the file type is unsupported or an error occurs.
    """
    documents = []
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        extension = file_path.suffix.lower()

        if extension == '.pdf':
            documents = load_pdf_document(file_path)
        elif extension == '.txt':
            documents = [load_txt_document(file_path)] # Wrap in list for consistency
        else:
            logger.warning(f"Unsupported file type for {file_path.name}: {extension}")
            # Optionally, you could raise an error here if unsupported files should halt execution
            # raise ValueError(f"Unsupported file type: {extension}")
            return [] # Return empty list for unsupported types
            
    except FileNotFoundError as fnfe:
        logger.error(f"Processing failed: {fnfe}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing {file_path.name}: {str(e)}")
    
    return documents
