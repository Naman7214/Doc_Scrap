import os

from dotenv import load_dotenv
from google import genai
from openai import AsyncOpenAI
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "document-index")

DIMENSION = 768
MAX_DEPTH = 4
MAX_LLM_REQUEST_COUNT = 50
CHUNK_SEMAPHORE_LIMIT = 30
MAX_CONCURRENT_TASKS = 20
MAX_CONCURRENT_CLICKS = 5

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
g_client = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, pool_threads=30)


# Adding the selector hierarchy from the second code
SELECTOR_HIERARCHY = [
    "button[role='tab']",
    "div[role='tab']",
    "[class*='data-lang']",
    "[class*='language-']",
    "[role='option']",
    "select",
    "option",
    "button, div, span, li",
]

PROGRAMMING_LANGUAGES = {
    "http",
    "python",
    "javascript",
    "typescript",
    "rust",
    "java",
    "csharp",
    "go",
    "curl",
    "json",
    "c#",
    "csharp",
    "node.js",
    "node",
    "npm",
    "yarn",
    "pnpm",
    "react",
    "angular",
    "vue",
    "svelte",
    "sql",
    "php",
    "ruby",
    "twilio-cli",
    "node",
    "cpp",
    ".net",
    "stripe-cli",
    "scala",
    "r",
}
