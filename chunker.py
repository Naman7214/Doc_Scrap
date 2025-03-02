import asyncio
import json
import os
import aiofiles
from typing import List, Dict
from .config import client, CHUNK_SEMAPHORE_LIMIT
from .filters import chunk_prompt
import openai
import time
import re

async def process_file(file_path, semaphore):
    log_dir = "chunk_usage_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir,os.path.basename(file_path).replace('.json', '_usage_log.json'))

    async with aiofiles.open(file_path, 'r') as file:
        data = json.loads(await file.read())

    tasks = [
        chunk_with_gpt(f"{chunk_prompt}\n**INPUT:**\n{item}\n**OUTPUT:**", log_file, semaphore)
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


async def chunk_with_gpt(text, log_file, chunk_semaphore):
    global chunk_llm_request_count, chunk_total_input_tokens, chunk_total_output_tokens
    print("Processing text ===========")
    
    async with chunk_semaphore:
        try:
            start_time = time.time()
            print("Sending request...")
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": text}],
                        temperature=0
                    ),
                    timeout = 90
                )
            except asyncio.TimeoutError:
                print("Request timed out.")
                return None
            end_time = time.time()
            print("Received response.")
        except openai.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return None

        chunk_llm_request_count += 1
        usage = getattr(response, "usage", None)
        if not usage:
            print("No usage info found in response.")
            return None

        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        chunk_total_input_tokens += input_tokens
        chunk_total_output_tokens += output_tokens

        # Prepare and write log data.
        log_data = {
            "llm_request_count": chunk_llm_request_count,
            "total_input_tokens": chunk_total_input_tokens,
            "total_output_tokens": chunk_total_output_tokens,
            "start_time": start_time,
            "end_time": end_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_taken": end_time - start_time
        }
        async with aiofiles.open(log_file, mode="a") as log_f:
            await log_f.write(json.dumps(log_data, indent=2) + "\n")

        output_text = response.choices[0].message.content.strip()
        chunks = extract_json_list(output_text)
        
        return chunks

def extract_json_list(text):
    # Regex to extract the JSON list from the response text.
    match = re.search(r'json\s*(\[.*?\])\s*', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # Convert string to a Python list
        except json.JSONDecodeError:
            return None  # Handle invalid JSON cases
    return None