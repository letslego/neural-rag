import os

from dotenv import load_dotenv

load_dotenv()

LM_MODEL = os.getenv("NEURAL_RAG_LM_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = os.getenv("NEURAL_RAG_DATA_DIR", "data/kgs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

DEFAULT_TOP_K = int(os.getenv("NEURAL_RAG_TOP_K", "5"))
GNN_LAYERS = int(os.getenv("NEURAL_RAG_GNN_LAYERS", "3"))
GNN_HIDDEN = int(os.getenv("NEURAL_RAG_GNN_HIDDEN", "256"))
SUBGRAPH_HOPS = int(os.getenv("NEURAL_RAG_SUBGRAPH_HOPS", "3"))
GNN_WEIGHTS = os.getenv("NEURAL_RAG_GNN_WEIGHTS", "")
