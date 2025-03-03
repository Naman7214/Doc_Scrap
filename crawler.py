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

queue = asyncio.Queue()
results = {}
processed_urls = set()
llm_request_counts = {}
count_locks = {}
log_lock = asyncio.Lock()
total_input_tokens = 0
total_output_tokens = 0

max_llm_request_count = MAX_LLM_REQUEST_COUNT


async def log_usage(start_time, end_time, input_tokens, output_tokens):
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
    
async def should_process_url(file_name):
    """Check if we should process more URLs for this file_name"""
    async with count_locks.get(file_name, asyncio.Lock()):
        return llm_request_counts.get(file_name, 0) < max_llm_request_count


async def filter_links_gpt(links, file_name):
    """Filter links using GPT-4o-mini asynchronously with minimal locking."""
    global llm_request_counts
    
    # If no links to filter, return empty list without making an LLM call
    if not links:
        return []
    
    # Check if we've already hit the limit - only lock for this short check
    async with count_locks.get(file_name, asyncio.Lock()):
        if llm_request_counts.get(file_name, 0) >= max_llm_request_count:
            return []
        # Increment preemptively to avoid race conditions
        llm_request_counts[file_name] = llm_request_counts.get(file_name, 0) + 1
    
    input_text = f"{filter_prompt}\n**INPUT:**\n{links}\n**OUTPUT:**"
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": input_text}],
            temperature=0
        )
        
        end_time = time.time()
        
        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        
        # Log usage asynchronously without blocking
        asyncio.create_task(log_usage(start_time, end_time, input_tokens, output_tokens))
        
        filtered_links = response.choices[0].message.content.strip()
        return clean_gpt_output(filtered_links)
        
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        # Release the counter if the call failed
        async with count_locks.get(file_name, asyncio.Lock()):
            llm_request_counts[file_name] = max(0, llm_request_counts.get(file_name, 0) - 1)
        return []


def remove_fragment(url):
    """Removes fragment identifiers (#) from URLs."""
    match = re.match(r"(https?://[^\s#]+)", url)
    return match.group(1) if match else url

def filter_urls_by_domain(base_url, url_list):
    """Filters URLs that belong to the same domain as the base URL."""
    base_domain = urlparse(base_url).netloc
    return [url for url in url_list if urlparse(url).netloc == base_domain]


async def worker(worker_id: int):
    """Worker coroutine that processes URLs from the queue concurrently."""
    while True:
        try:
            url, depth, file_name = await queue.get()
            
            await crawl_page(url, depth, file_name)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WORKER ERROR] Worker {worker_id}: {e}")
        finally:
            queue.task_done()

async def crawl_page(url: str, depth: int, file_name):
    """
    Scrape a single URL using Crawl4AI, extract internal links, and process markdown.
    """
    global results, queue, processed_urls
    
    # First check if we should process this URL
    if not await should_process_url(file_name):
        return
    
    if depth >= MAX_DEPTH:
        return
    
    print(f"[CRAWL] Processing {url} at depth {depth}")
    
    try:
        prune_filter = PruningContentFilter(
            threshold_type="dynamic",
        )
        md_generator = DefaultMarkdownGenerator(
            content_filter=prune_filter,
            options={
                "ignore_links": True,
                "escape_html": True,
                "ignore_images": True,
                "skip_internal_links": True,
            },
        )
        crawler_cfg = CrawlerRunConfig(
            exclude_external_links=True,
            exclude_social_media_links=True,
            exclude_external_images=True,
            verbose=False,
            cache_mode=CacheMode.DISABLED,
            markdown_generator=md_generator,
            page_timeout=10000,
        )
        browser_conf = BrowserConfig(text_mode=True, light_mode=True, verbose=False)
        
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=url, config=crawler_cfg)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        # Remove from pending to allow other URLs to be processed
        return
    
    if not result.success:
        print(f"[FAILED] Crawling unsuccessful for {url}")
        return
        
    # Store the result
    if file_name not in results:
        results[file_name] = []
    # hidden_snippets = await extract_hidden_snippets(url)
    # merged_content = merge_content(result.fit_markdown, hidden_snippets)
    # results[file_name].append({"href": url, "content": merged_content})
    results[file_name].append({"href": url, "content": result.fit_markdown})

    
    # Only continue if we haven't reached the LLM limit
    if not await should_process_url(file_name):
        return
    
    # Extract and filter internal links efficiently
    internal_links = [
        remove_fragment(x["href"]) for x in result.links.get("internal", [])
    ]
    internal_links = list(set(internal_links))
    # Apply domain filter first to reduce the number of URLs sent to GPT
    internal_links = filter_urls_by_domain(url, internal_links)
    
    # Filter links using GPT
    filtered_links = await filter_links_gpt(internal_links, file_name)
    
    # Efficiently process new links
    if await should_process_url(file_name):
        new_links = []
        for link in filtered_links:
            # Check if we've processed or are pending on this URL
            if link not in processed_urls :
                processed_urls.add(link)
                new_links.append((link, depth + 1, file_name))
        
        # Add all new links to queue at once
        for link_info in new_links:
            await queue.put(link_info)
    
    # Mark as no longer pending

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
