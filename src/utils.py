import pickle
import logging
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from config import EMBEDDING_MODEL, VECTOR_STORE_DIR, CHUNKS_PATH
from data_loaders import load_company_info, load_faq_documents
from text_processor import markdown_splitter, recursive_500

logger = logging.getLogger(__name__)

def load_company_vector_store() -> Optional[FAISS]:
    """Load existing vector store with proper error handling"""
    try:
        if Path(VECTOR_STORE_DIR).exists():
            vector_store = FAISS.load_local(
                str(VECTOR_STORE_DIR), 
                EMBEDDING_MODEL, 
                allow_dangerous_deserialization=True
            )
            logger.info("Successfully loaded existing vector store")
            return vector_store
        else:
            logger.info("No existing vector store found")
            return None
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None

def create_company_vector_store(documents: List[Document]) -> Optional[FAISS]:
    """Create and save vector store with error handling"""
    if not documents:
        logger.error("No documents provided to create vector store")
        return None
        
    try:
        # Ensure directory exists
        Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
        
        vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
        vector_store.save_local(str(VECTOR_STORE_DIR))
        logger.info(f"Successfully created and saved vector store with {len(documents)} documents")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        return None

def create_company_documents() -> List[Document]:
    """Create company documents with error handling"""
    try:
        company_documents = []
        
        # Load FAQ documents
        try:
            faq_docs = load_faq_documents()
            company_documents.extend(faq_docs)
            logger.info(f"Loaded {len(faq_docs)} FAQ documents")
        except Exception as e:
            logger.error(f"Failed to load FAQ documents: {e}")
        
        # Load company info
        try:
            company_info = load_company_info()
            if company_info:
                company_documents.append(company_info)
                logger.info("Loaded company info document")
        except Exception as e:
            logger.error(f"Failed to load company info: {e}")
        
        logger.info(f"Total documents loaded: {len(company_documents)}")
        return company_documents
        
    except Exception as e:
        logger.error(f"Failed to create company documents: {e}")
        return []

def split_documents(company_documents: List[Document]) -> List[Document]:
    """Split documents into chunks with error handling"""
    if not company_documents:
        logger.warning("No documents provided for splitting")
        return []
        
    company_chunks = []
    
    try:
        for i, doc in enumerate(company_documents):
            try:
                if doc.metadata.get("type") == "general_info":
                    # Use markdown splitter for info.md
                    split_docs = markdown_splitter.split_text(doc.page_content)
                    for d in split_docs:
                        d.metadata.update(doc.metadata)
                    company_chunks.extend(split_docs)
                    logger.debug(f"Split document {i} using markdown splitter")
                else:
                    # Use recursive splitter for FAQs
                    split_docs = recursive_500.split_documents([doc])
                    company_chunks.extend(split_docs)
                    logger.debug(f"Split document {i} using recursive splitter")
                    
            except Exception as e:
                logger.error(f"Failed to split document {i}: {e}")
                continue
                
        logger.info(f"Successfully split {len(company_documents)} documents into {len(company_chunks)} chunks")
        return company_chunks
        
    except Exception as e:
        logger.error(f"Failed to split documents: {e}")
        return []

def load_chunks() -> Optional[List[Document]]:
    """Load pre-processed chunks with error handling"""
    try:
        if Path(CHUNKS_PATH).exists():
            with open(CHUNKS_PATH, 'rb') as f:
                company_chunks = pickle.load(f)
            logger.info(f"Successfully loaded {len(company_chunks)} chunks from cache")
            return company_chunks
        else:
            logger.info("No cached chunks found")
            return None
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        return None

def save_chunks(chunks: List[Document]) -> bool:
    """Save processed chunks to file"""
    try:
        # Ensure directory exists
        Path(CHUNKS_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"Successfully saved {len(chunks)} chunks to {CHUNKS_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to save chunks: {e}")
        return False

def initialize_knowledge_base() -> Optional[FAISS]:
    """Initialize the complete knowledge base"""
    try:
        # Try to load existing vector store
        vector_store = load_company_vector_store()
        if vector_store:
            return vector_store
        
        # If no existing store, create new one
        logger.info("Creating new knowledge base...")
        
        # Load or create chunks
        chunks = load_chunks()
        if not chunks:
            logger.info("No cached chunks found, processing documents...")
            documents = create_company_documents()
            if documents:
                chunks = split_documents(documents)
                if chunks:
                    save_chunks(chunks)
        
        if chunks:
            vector_store = create_company_vector_store(chunks)
            return vector_store
        else:
            logger.error("No chunks available to create vector store")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        return None