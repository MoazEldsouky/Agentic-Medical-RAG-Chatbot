import os
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, VECTOR_STORE_DIR
from data_loader import load_company_info, load_faq_documents
from text_processor import markdown_splitter, recursive_500
import logging

logger = logging.getLogger(__name__)

# Function to load company vector store
def load_company_vector_store():
    if os.path.exists(VECTOR_STORE_DIR):
        vector_store = FAISS.load_local(VECTOR_STORE_DIR, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None

# Function to create company vector store
def create_company_vector_store(documents: List[Document]):
    vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
    vector_store.save_local(VECTOR_STORE_DIR)
    return vector_store


# Function to create documents from company info
def create_company_documents():
    company_documents = load_faq_documents()
    company_documents.append(load_company_info())
    return company_documents

# Function to split documents to chunks
def split_documents(company_documents: List[Document]) -> List[Document]:
    company_chunks = []

    for doc in company_documents:
        if doc.metadata.get("type") == "general_info":
            # Use markdown splitter for info.md
            split_docs = markdown_splitter.split_text(doc.page_content)
            for d in split_docs:
                d.metadata.update(doc.metadata)
            company_chunks.extend(split_docs)
        else:
            # Use recursive splitter for FAQs
            split_docs = recursive_500.split_documents([doc])
            company_chunks.extend(split_docs)
            
    return company_chunks

# 