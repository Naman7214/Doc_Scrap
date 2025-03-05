import asyncio
import json
import os

import aiofiles

from filters import chunk_prompt, summary_links_prompt, summary_prompt
from utils.chunking_utils import (
    chunk_with_gpt,
    extract_hrefs,
    fetch_content,
    filter_links_gpt,
    generate_summary_chunk,
)

chunk_llm_request_count = 0
chunk_total_input_tokens = 0
chunk_total_output_tokens = 0


async def process_file(file_path, semaphore):
    log_dir = "chunk_usage_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, os.path.basename(file_path).replace(".json", "_usage_log.json")
    )

    async with aiofiles.open(file_path, "r") as file:
        data = json.loads(await file.read())

    tasks = [
        chunk_with_gpt(
            f"{chunk_prompt}\n**INPUT:**\n{item}\n**OUTPUT:**",
            log_file,
            semaphore,
        )
        for item in data
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    final_chunks = []
    for response in responses:
        if isinstance(response, Exception):
            print(f"Task failed with exception: {response}")
        elif response is not None:
            final_chunks.extend(response)

    return final_chunks


async def process_summary_file(file_path):
    log_dir = "chunk_usage_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, os.path.basename(file_path).replace(".json", "_usage_log.json")
    )

    async with aiofiles.open(file_path, "r") as file:
        data = json.loads(await file.read())

    links = extract_hrefs(data)
    if len(links) > 180:
        links = links[:180]
    filtered_links = await filter_links_gpt(
        f"{summary_links_prompt}\n**INPUT:**\n{links}\n**OUTPUT:**", log_file
    )

    content_data = fetch_content(data, filtered_links)
    responses = await generate_summary_chunk(
        f"{summary_prompt}\n**INPUT:**\n{content_data}\n**OUTPUT:**", log_file
    )

    return responses
