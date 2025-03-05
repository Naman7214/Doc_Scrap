import asyncio
import concurrent.futures
import json

from utils.embedding_utils import get_embedding, get_sparse_embedding

request_count = 0


async def get_embedding_concurrently(text, pool, semaphore):
    async with semaphore:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(pool, get_embedding, text)
        return result


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
        tasks.append(
            asyncio.create_task(
                get_embedding_concurrently(chunk_text, pool, semaphore)
            )
        )

    embeddings = await asyncio.gather(*tasks)

    for item, embedding in zip(data, embeddings):
        item["embedding"] = embedding
        item["sparse_values"] = get_sparse_embedding(item["chunked_data"])
        embedded_count += 1
        print(
            f"Embedded {embedded_count}/{total_chunks} chunks in {source_file}"
        )

    with open(source_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Embeddings added and saved successfully for {source_file}.")


async def process_files(file_list, max_concurrent_tasks):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        tasks = [embed_process_file(file_list, pool, semaphore)]
        await asyncio.gather(*tasks)
