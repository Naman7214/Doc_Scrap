import json
import re
import aiofiles
from config import client, CHUNK_SEMAPHORE_LIMIT
import time
import openai
import asyncio


chunk_llm_request_count = 0
chunk_total_input_tokens = 0
chunk_total_output_tokens = 0

def extract_hrefs(json_data):
    hrefs = []
    for entry in json_data:
        href = entry.get('href', '')
        hrefs.append(href)
    return hrefs

def fetch_content(json_data, hrefs):
    content_dict = {}
    for entry in json_data:
        href = entry.get('href', '')
        if href in hrefs:
            content = entry.get('content', '')
            content_dict[href] = content
    return content_dict

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
    match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # Convert string to a Python list
        except json.JSONDecodeError:
            return None  # Handle invalid JSON cases
    return None


async def filter_links_gpt(text, log_file):
    global chunk_llm_request_count, chunk_total_input_tokens, chunk_total_output_tokens
    print("Processing Links ===========")
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
    filtered_links = extract_json_list(output_text)
    
    return filtered_links


async def generate_summary_chunk(text,log_file):
    global chunk_llm_request_count, chunk_total_input_tokens, chunk_total_output_tokens
    print("Processing Summary ===========")
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
                timeout = 900
            )
            print("Summary Chunk Generated")
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