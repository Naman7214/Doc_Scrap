import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import re
from urllib.parse import urlparse
import time
from filters import filter_prompt
from openai import Client
from typing import List, Dict
from config import (
    client, OPENAI_API_KEY, GEMINI_API_KEY, pc,
    MAX_DEPTH, MAX_LLM_REQUEST_COUNT
)
import aiofiles
import json
import os

total_input_tokens = 0
total_output_tokens = 0
log_lock = asyncio.Lock()


async def get_file_name(base_url):
    try:
        browser_conf = BrowserConfig(text_mode=True, light_mode=True, verbose=False)
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=base_url)
            title = result.metadata["title"]
            clean_title = re.sub(r'[^\w\s]', '', title)  # Remove special characters
            clean_title = re.sub(r'\s+', '_', clean_title)  # Replace spaces with underscores
            return clean_title
    except Exception as e:
        print(f"[ERROR] Failed to get title for {base_url}: {e}")
        return urlparse(base_url).netloc.replace(".", "_")
    
    
    
def remove_fragment(url):
    """Removes fragment identifiers (#) from URLs."""
    match = re.match(r"(https?://[^\s#]+)", url)
    return match.group(1) if match else url

def filter_urls_by_domain(base_url, url_list):
    """Filters URLs that belong to the same domain as the base URL."""
    base_domain = urlparse(base_url).netloc
    return [url for url in url_list if urlparse(url).netloc == base_domain]


async def log_usage(start_time, end_time, input_tokens, output_tokens, llm_request_counts):
    """Log token usage asynchronously but with minimal locking"""
    global total_input_tokens, total_output_tokens
    
    # Update counters atomically
    async with log_lock:
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        combined_llm_request_count = sum(llm_request_counts.values())
        
        # Prepare log data
        log_data = {
            "timestamp": time.time(),
            "request_count": combined_llm_request_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "time_taken": end_time - start_time,
        }
    
    # Write to file outside the lock
    async with aiofiles.open("llm_usage_log.json", mode="a") as log_file:
        await log_file.write(json.dumps(log_data) + "\n")

async def save_results(results: dict, directory: str = "results"):
    """
    Saves the given results in separate JSON files inside the specified directory.
    Each key in the results dictionary becomes a JSON filename.
    """
    os.makedirs(directory, exist_ok=True)
    
    # Create tasks for all file saves to run in parallel
    save_tasks = []
    for filename, data in results.items():
        file_path = os.path.join(directory, f"{filename}.json")
        
        async def save_file(path, content):
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(content, ensure_ascii=False, indent=2))
            print(f"Saved: {path}")
            
        save_tasks.append(save_file(file_path, data))
    
    # Run all save tasks concurrently
    await asyncio.gather(*save_tasks)

def clean_gpt_output(response_text):
    """Cleans GPT output by removing code block markers and ensuring a valid list format."""
    response_text = re.sub(r"```[a-zA-Z]*", "", response_text).strip("`").strip()
    try:
        return eval(response_text)
    except:
        # Fallback if the output isn't a valid Python list
        print(f"Warning: Could not parse GPT output - {response_text}")
        return []
    

def merge_content(markdown_content, hidden_snippets):
    """Merges extracted markdown content with hidden code snippets."""
    # Regular expression to identify code blocks (```language ... ```)
    code_block_pattern = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)

    merged_content = ""
    last_end = 0
    inserted_languages = set()

    for match in code_block_pattern.finditer(markdown_content):
        language = match.group(1).lower()
        code = match.group(2)

        # Append the markdown content before the current code block
        merged_content += markdown_content[last_end:match.start()]
        
        # Append the default extracted code
        merged_content += f"```{language}\n{code}\n```\n"

        # Append hidden snippets for other languages after the default language snippet
        if language in hidden_snippets:
            for alt_code in hidden_snippets.pop(language, []):
                merged_content += f"\n```{language}\n{alt_code}\n```\n"
            inserted_languages.add(language)

        last_end = match.end()

    # Append any remaining content
    merged_content += markdown_content[last_end:]

    # If there are remaining hidden snippets, append them at the end
    if hidden_snippets:
        merged_content += "\n\n# Additional Code Snippets\n"
        for lang, snippets in hidden_snippets.items():
            if lang not in inserted_languages:
                for snippet in snippets:
                    merged_content += f"\n```{lang}\n{snippet}\n```\n"

    return merged_content