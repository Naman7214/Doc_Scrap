import asyncio
import json
import os

import aiofiles
from playwright.async_api import async_playwright

from chunker import process_file, process_summary_file
from config import (
    CHUNK_SEMAPHORE_LIMIT,
    INDEX_NAME,
    MAX_CONCURRENT_TASKS,
    MAX_LLM_REQUEST_COUNT,
    pc,
)
from crawler import processed_urls, queue, worker_for_full_page, llm_request_counts, count_locks, results, code_snippets_crawler
from embedding import process_files
from urls import start_urls
from utils.crawler_utils import get_file_name, save_results
from utils.pinecone_utils import (
    ensure_index_exists,
    load_json_files_for_pinecone,
    pine_chunks,
)

max_llm_request_count = MAX_LLM_REQUEST_COUNT
max_concurrent_tasks = MAX_CONCURRENT_TASKS


async def main(start_urls: list[str], num_workers: int = 60):
    global results, llm_request_counts, count_locks

    # Initialize locks more efficiently
    file_names = []

    # Create tasks to get file names concurrently
    file_name_tasks = [get_file_name(url) for url in start_urls]
    file_names = await asyncio.gather(*file_name_tasks)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        # Initialize tracking
        for i, url in enumerate(start_urls):
            file_name = file_names[i]
            count_locks[file_name] = asyncio.Lock()
            results[file_name] = []
            llm_request_counts[file_name] = 0

            # Add to queue
            processed_urls.add(url)
            await queue.put((url, 1, file_name))
            print(f"Starting with base URL: {url} -> {file_name}")

        # Create worker tasks
        tasks = [
            asyncio.create_task(worker_for_full_page(i)) for i in range(num_workers)
        ]

        # Wait for all queue tasks to be processed
        await queue.join()

        # Cancel all worker tasks
        for task in tasks:
            task.cancel()
            
        # comment below code to ignore the hidden code snippets.
        results = await  code_snippets_crawler( num_workers= 25, results= results, browser= browser)
        # Save results
        await save_results(results)

        # Wait for tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()

    # Print summary
    print("\n--- CRAWL SUMMARY ---")
    for file_name, count in llm_request_counts.items():
        print(
            f"{file_name}: {count}/{max_llm_request_count} LLM calls, {len(results.get(file_name, []))} pages crawled"
        )
    # ================================================================================================
    # ================================= CHUNKING ====================================================
    # ================================================================================================

    json_files = [
        os.path.join("results", file) for file in os.listdir("results")
    ]

    all_chunks = []
    semaphore = asyncio.Semaphore(CHUNK_SEMAPHORE_LIMIT)

    for file in json_files:
        chunks = await process_file(file, semaphore)
        all_chunks.extend(chunks)
        summary_chunks = await process_summary_file(file)
        all_chunks.extend(summary_chunks)

    chunk_dir = os.path.join("json_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_file = os.path.join(chunk_dir, "all_chunks.json")

    async with aiofiles.open(chunk_file, mode="w") as chunk_f:
        await chunk_f.write(json.dumps(all_chunks, indent=2))

    print("Saved all chunks to", chunk_file)

    # ================================================================================================
    # ================================= EMBEDDING ====================================================
    # ================================================================================================

    chunk_json_path = os.path.join("json_chunks", "all_chunks.json")
    print(f"Processing embeddings for {chunk_json_path}...")
    await process_files(chunk_json_path, max_concurrent_tasks)

    # ================================================================================================
    # ================================= PINECONE ====================================================
    # ================================================================================================

    embeddings_folder = "json_chunks"
    vector_data = load_json_files_for_pinecone(directory_path=embeddings_folder)

    if not vector_data:
        print("No data found to upload. Exiting.")
        return

    DIMENSION = len(vector_data[0]["values"])
    print(f"Using dimension {DIMENSION} from first vector")

    # Ensure index exists
    if not ensure_index_exists(INDEX_NAME, DIMENSION):
        print("Failed to create or validate index. Exiting.")
        return

    # Connect to the index
    index = pc.Index(name=INDEX_NAME)

    # Check index stats before upload
    before_stats = index.describe_index_stats()
    print(f"Index stats before upload: {before_stats}")

    print(f"Starting upload of {len(vector_data)} vectors in batches...")
    # Perform batched async upserts
    try:
        async_results = [
            index.upsert(vectors=batch, async_req=True, namespace="default")
            for batch in pine_chunks(
                vector_data, batch_size=200
            )  # Decreased batch size for reliability
        ]

        # Wait for completion and handle results
        for i, async_result in enumerate(async_results):
            try:
                result = async_result.result()  # Use result() instead of get()
                print(f"Batch {i+1}/{len(async_results)} completed: {result}")
            except Exception as e:
                print(f"Error in batch {i+1}: {str(e)}")

        print("Upserts completed successfully!")

    except Exception as e:
        print(f"Error during upsert process: {str(e)}")


if __name__ == "__main__":

    asyncio.run(main(start_urls))
