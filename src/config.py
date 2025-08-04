from warnings import filterwarnings
filterwarnings("ignore")
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from huggingface_hub import login
from dotenv import load_dotenv

# Initialize environment
load_dotenv()
login(os.getenv("HUGGINGFACE_HUB_TOKEN"))


# --- File Path Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
COMPANY_INFO_DIR = os.path.join(DATA_DIR, "company_info")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector_store")

# --- LLM Configuration ---
LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)