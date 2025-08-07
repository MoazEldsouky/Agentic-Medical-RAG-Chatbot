from utils import *

# Try to load existing vector store, create if not found
try:
    print("Loading vector store...")
    vector_store = load_company_vector_store()
    if vector_store:
        print("Vector store loaded successfully")
    else:
        # If vector_store is None, this means it didn't exist
        print("Vector store not found, creating new...")
        company_documents = create_company_documents()
        company_chunks = split_documents(company_documents)
        vector_store = create_company_vector_store(company_chunks)
        print("Vector store created successfully")
except Exception as e:
    # This block will handle other potential errors during the loading/creation process
    logger.error(f"Error loading or creating vector store: {str(e)}")
    # It might be good to exit or handle this more gracefully.
    # For now, let's just re-raise the exception to see what's wrong.
    raise


