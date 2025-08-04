# Import required libraries
import pandas as pd
from config import COMPANY_INFO_DIR
from pathlib import Path
from typing import List
from langchain.schema import Document
import logging
logger = logging.getLogger(__name__)

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
                }
            )
            documents.append(doc)
            
        logger.info(f"Loaded {len(documents)} FAQ documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading FAQ documents: {str(e)}")
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
        logger.info("Loaded company info document")
        return doc
        
    except Exception as e:
        logger.error(f"Error loading company info: {str(e)}")
        raise


