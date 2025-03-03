import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone

from config import INDEX_NAME, pc
from embedding import get_embedding

load_dotenv()
from judge_prompt import judge_prompt
from questions import queries

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# Connect to the Pinecone index that holds the document chunks
index_name = INDEX_NAME
index = pc.Index(index_name)


def rerank(query, chunks):
    result = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=query,
        documents=chunks,
        top_n=10,
        return_documents=True,
        parameters={"truncate": "END"},
    )

    reranked_chunks = [entry["document"]["text"] for entry in result.data]

    return reranked_chunks


total_score = 0
ratings = []

# Iterate over each query to retrieve, rerank, and judge relevancy
for query in queries:
    query_embedding = get_embedding(text=query)
    # 1. Retrieve similar chunks from Pinecone based on the query
    result = index.query(
        vector=query_embedding,
        top_k=20,
        include_metadata=True,
        namespace="default",
        include_values=False,
    )

    retrieved_chunks = [
        match["metadata"]["chunked_data"] for match in result["matches"]
    ]

    # 2. Rerank the retrieved chunks using the reranker available in Pinecone
    reranked_chunks = rerank(query, retrieved_chunks)
    context_str = "\n".join(reranked_chunks)

    # 3. Build the prompt for the judge LLM (gpt-4o-mini)
    prompt = f"{judge_prompt}\n**QUERY**:\n{query}\n**CONTEXT**\n{context_str}"

    # 4. Call OpenAI's API with the prompt to get the relevancy score
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a judge evaluating the relevancy of provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=5,
        temperature=0.0,
    )

    # Extract the rating (expecting a number between 1 and 5)
    rating_text = response.choices[0].message.content.strip()
    print(f"========={rating_text}")
    try:
        rating = int(rating_text)
    except ValueError:
        rating = 0  # Default to 0 if parsing fails

    ratings.append(rating)
    total_score += rating

# 5. Calculate and print the average relevancy score
average_relevancy = total_score / len(queries) if queries else 0
print("Average relevancy score:", average_relevancy)
