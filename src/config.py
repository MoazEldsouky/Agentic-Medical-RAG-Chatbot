import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from huggingface_hub import login
from dotenv import load_dotenv
import logging

# Initialize environment
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Secure HuggingFace login with error handling
try:
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(hf_token)
        logger.info("Successfully logged into HuggingFace")
    else:
        logger.warning("No HuggingFace token found - some features may be limited")
except Exception as e:
    logger.error(f"HuggingFace login failed: {e}")

# --- File Path Configuration (Cross-platform compatible) ---
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
COMPANY_INFO_DIR = DATA_DIR / "raw_company_info"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "company_chunks.pkl"
VECTOR_STORE_DIR = PROCESSED_DIR / "vector_store"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
COMPANY_INFO_DIR.mkdir(parents=True, exist_ok=True)

# --- LLM Configuration with error handling ---
def create_llm():
    """Create LLM with proper error handling and fallbacks"""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable.")
    
    try:
        return ChatOpenAI(
            model="gpt-4o",  
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL"),  # Optional custom endpoint
            temperature=0.0,
            max_tokens=1024,
            request_timeout=30,  # Increased timeout for stability
            max_retries=2,
            streaming=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise

LLM = create_llm()

# --- Embedding Model Configuration with error handling ---
def create_embedding_model():
    """Create embedding model with proper error handling"""
    try:
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        # Fallback to a simpler model
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e2:
            logger.error(f"Fallback embedding model also failed: {e2}")
            raise

EMBEDDING_MODEL = create_embedding_model()

# Configuration validation
def validate_config():
    """Validate all required configurations"""
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Check if data directories exist
    if not COMPANY_INFO_DIR.exists():
        logger.warning(f"Company info directory not found: {COMPANY_INFO_DIR}")
    
    logger.info("Configuration validation completed")

# Run validation on import
try:
    validate_config()
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    raise e
