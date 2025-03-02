import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google import genai
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()  

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "DocumentIndex")

DIMENSION = 768
MAX_DEPTH = 4
MAX_LLM_REQUEST_COUNT = 50
CHUNK_SEMAPHORE_LIMIT = 30
MAX_CONCURRENT_TASKS = 1

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
g_client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, pool_threads=30)