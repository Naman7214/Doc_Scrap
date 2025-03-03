import asyncio
import concurrent.futures
import json
from config import g_client
from fastembed import TextEmbedding
import types
from pinecone_text.sparse import BM25Encoder


model = TextEmbedding("BAAI/bge-base-en-v1.5")
bm25 = BM25Encoder().default()

request_count = 0
async def get_embedding_concurrently(text, pool, semaphore):
    async with semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(pool, get_embedding, text)
        return result

def get_sparse_embedding(text):
    doc_sparse_vector = bm25.encode_documents(text)
    return {
        "indices": doc_sparse_vector["indices"],  
        "values": doc_sparse_vector["values"]
    }

def get_embedding(text):
    global request_count
    request_count += 1
    print(f"Embedding text. Request count: {request_count}")
    try:
        # Get the raw embedding output, and force conversion from generator to list.
        raw = model.embed(text, batch_size=24, parallel=True)
        # If raw is a generator, exhaust it.
        if isinstance(raw, types.GeneratorType):
            raw = list(raw)
        
        # Now, if raw is a numpy array, convert it to a list.
        if hasattr(raw, 'tolist'):
            embeddings = raw.tolist()
        # Otherwise, if it's already a list, check each element.
        elif isinstance(raw, list):
            embeddings = [e.tolist() if hasattr(e, 'tolist') else e for e in raw]
        else:
            embeddings = raw

        print("Received response")
        return embeddings[0]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


async def embed_process_file(source_file, pool, semaphore):
    print("Processing file")
    print(source_file)
    with open(source_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    total_chunks = len(data)
    embedded_count = 0

    tasks = []
    for item in data:
        chunk_text = item["chunked_data"]
        tasks.append(asyncio.create_task(get_embedding_concurrently(chunk_text, pool, semaphore)))

    embeddings = await asyncio.gather(*tasks)

    for item, embedding in zip(data, embeddings):
        item["embedding"] = embedding
        item["sparse_values"] = get_sparse_embedding(item["chunked_data"])
        embedded_count += 1
        print(f"Embedded {embedded_count}/{total_chunks} chunks in {source_file}")

    with open(source_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Embeddings added and saved successfully for {source_file}.")



async def process_files(file_list, max_concurrent_tasks):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        tasks = [embed_process_file(file_list, pool, semaphore)]
        await asyncio.gather(*tasks)