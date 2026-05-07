import os

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("NEURAL_RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("NEURAL_RAG_CHROMA_PATH", ".chroma")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEFAULT_TOP_K = int(os.getenv("NEURAL_RAG_TOP_K", "5"))
