from utils import *
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config import logger

# Try to load existing vector store, create if not found
try:
    logger.info("Loading vector store...")
    vector_store = load_company_vector_store()
    if vector_store:
        logger.info("Vector store loaded successfully")
    else:
        # If vector_store is None, this means it didn't exist
        logger.info("Vector store not found, creating new...")
        company_documents = create_company_documents()
        company_chunks = split_documents(company_documents)
        vector_store = create_company_vector_store(company_chunks)
        logger.info("Vector store created successfully")
except Exception as e:
    # This block will handle other potential errors during the loading/creation process
    logger.error(f"Error loading or creating vector store: {str(e)}")
    # It might be good to exit or handle this more gracefully.
    # For now, let's just re-raise the exception to see what's wrong.
    raise


# Try to load existing company chunks, create if not found
try:
    logger.info("Loading company chunks...")
    company_chunks = load_chunks()
    if company_chunks:
        logger.info("Company chunks loaded successfully")
    else:
        # If company_chunks is None, this means it didn't exist
        logger.info("Company chunks not found, creating new...")
        company_documents = create_company_documents()
        company_chunks = split_documents(company_documents)
        save_chunks(company_chunks)
        logger.info("Company chunks created successfully")
        
except Exception as e:
    # This block will handle other potential errors during the loading/creation process
    logger.error(f"Error loading or creating company chunks: {str(e)}")
    # It might be good to exit or handle this more gracefully.
    # For now, let's just re-raise the exception to see what's wrong.
    raise


# Create vector retriever
logger.info("🔍 Creating vector retriever...")
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create BM25 retriever
logger.info("📝 Creating BM25 retriever...")
bm25_retriever = BM25Retriever.from_documents(company_chunks)
bm25_retriever.k = 3

# Create hybrid retriever
logger.info("🔄 Creating hybrid retriever...")
hybrid_retriever = EnsembleRetriever(
       retrievers=[bm25_retriever, vector_retriever],
       weights=[0.2, 0.8]
   )


logger.info("✅ Retrievers created and hybrid retriever is ready.")
