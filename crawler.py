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
    MAX_DEPTH, MAX_LLM_REQUEST_COUNT, MAX_CONCURRENT_CLICKS, SELECTOR_HIERARCHY, PROGRAMMING_LANGUAGES
)
import aiofiles
import json
import os
from utils.crawler_utils import  remove_fragment, filter_urls_by_domain, log_usage, merge_content, clean_gpt_output
queue = asyncio.Queue()
results = {}
processed_urls = set()
llm_request_counts = {}
count_locks = {}


max_llm_request_count = MAX_LLM_REQUEST_COUNT



async def handle_element_and_extract(page, element, text, seen_code_blocks, should_click=True):
    """
    Handle an element (click if needed) and extract code snippets from the page.
    
    :param page: The page object.
    :param element: The element to interact with.
    :param text: The text associated with the element (e.g., programming language name).
    :param seen_code_blocks: A set to track already processed code snippets.
    :param should_click: Whether to click the element before extracting code.
    :return: A tuple of (snippets, text).
    """
    snippets = []
    try:
        # Click the element if required
        if should_click:
            print(f"Clicking: {text} in element")
            await element.click()
            await asyncio.sleep(0.5)  # Reduced sleep time

        # Extract code blocks after the action
        code_blocks = await page.locator("pre code, pre, code, div[class*='bg-'] pre code, div[class*='bg-'] pre").all()
        for code_block in code_blocks:
            try:
                code_text = await code_block.inner_text(timeout=3000)
                code_text = code_text.strip()
                if code_text and code_text not in seen_code_blocks:
                    seen_code_blocks.add(code_text)
                    snippets.append(code_text)
            except Exception as e:
                print(e)
                continue
        return snippets, text
    except Exception as e:
        print(f"Skipping interactive element due to error: {e}")
        return [], text

async def extract_hidden_snippets(url, browser):
    """Extracts hidden code snippets by clicking on tabs and handling non-interactive content."""
    code_snippets = {}  # Store extracted snippets by language
    seen_code_blocks = set()
    
    context = await browser.new_context(accept_downloads = False)
    page = await context.new_page()
    await page.goto(url  = url, timeout = 45000)
    
    # Step 1: Use improved selector hierarchy to find relevant elements
    for selector in SELECTOR_HIERARCHY:
        try:
            elements = await page.locator(selector).all()
            if not elements:
                continue

            # Process elements concurrently
            click_tasks = []
            for element in elements:
                # Skip if the element is not visible
                if not await element.is_visible():
                    continue

                # Handle select elements differently
                if selector == "select":
                    # Locate the option elements within the select
                    options = await element.locator("option").all()
                    for option in options:
                        option_text = await option.inner_text(timeout=3000)
                        option_text = option_text.strip().lower()
                        if option_text in PROGRAMMING_LANGUAGES:
                            # Use select_option instead of clicking
                            value = await option.get_attribute("value")
                            await element.select_option(value=value)
                            # Extract code after selecting the option
                            click_tasks.append(handle_element_and_extract(page, element, option_text, seen_code_blocks, should_click=False))
                else:
                    # For non-select elements, check if the element text is in PROGRAMMING_LANGUAGES
                    element_text = await element.inner_text(timeout=3000)
                    element_text = element_text.strip().lower()
                    if element_text in PROGRAMMING_LANGUAGES:
                        # Proceed with the click logic
                        click_tasks.append(handle_element_and_extract(page, element, element_text, seen_code_blocks, should_click=True))

            # Execute click operations concurrently with a limit
            results = []
            for i in range(0, len(click_tasks), MAX_CONCURRENT_CLICKS):
                batch = click_tasks[i:i+MAX_CONCURRENT_CLICKS]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)

            # Process results
            for snippets, lang in results:
                if snippets and lang in PROGRAMMING_LANGUAGES:
                    code_snippets.setdefault(lang, []).extend(snippets)

        except Exception as e:
            print(f"Error with selector {selector}: {e}")

    # Step 2: Extract non-interactive hidden content
    hidden_elements = await page.query_selector_all("[style*='display: none'], [style*='visibility: hidden']")
    for element in hidden_elements:
        try:
            await page.evaluate("el => el.style.display = 'block'", element)  # Force show hidden elements
            text = await element.inner_text()
        except Exception as e:
            print(f"Skipping hidden element: {e}")

    # Step 3: Dynamically detect programming languages from code blocks
    languages = await page.evaluate("""() => {
        return Array.from(document.querySelectorAll('[class*="language-"]')).map(el => {
            const match = el.className.match(/language-(\w+)/);
            return match ? match[1] : null;
        }).filter(Boolean);
    }""")

    if languages:
        for lang in languages:
            if lang not in code_snippets:
                code_snippets[lang] = []

    await page.close()
    await context.close()
    return code_snippets


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
        asyncio.create_task(log_usage(start_time, end_time, input_tokens, output_tokens, llm_request_counts))
        
        filtered_links = response.choices[0].message.content.strip()
        return clean_gpt_output(filtered_links)
        
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        # Release the counter if the call failed
        async with count_locks.get(file_name, asyncio.Lock()):
            llm_request_counts[file_name] = max(0, llm_request_counts.get(file_name, 0) - 1)
        return []

async def worker(worker_id: int, browser):
    """Worker coroutine that processes URLs from the queue concurrently."""
    while True:
        try:
            url, depth, file_name = await queue.get()
            
            await crawl_page(url, depth, file_name, browser)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WORKER ERROR] Worker {worker_id}: {e}")
        finally:
            queue.task_done()

async def crawl_page(url: str, depth: int, file_name, browser):
    """
    Scrape a single URL using Crawl4AI, extract internal links, and process markdown.
    """
    global results, queue, processed_urls
    
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
            markdown_generator=md_generator
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
    
    hidden_snippets = await extract_hidden_snippets(url= url, browser= browser)
    md_content = result.fit_markdown
    final_md_content = merge_content(md_content,hidden_snippets )
    results[file_name].append({"href": url, "content": final_md_content })

    # Only continue if we haven't reached the LLM limit
    if not await should_process_url(file_name):
        return
    
    # Extract and filter internal links efficiently
    internal_links = list(set([
        remove_fragment(x["href"]) for x in result.links.get("internal", [])
    ]))
    # Apply domain filter first to reduce the number of URLs sent to GPT
    internal_links = filter_urls_by_domain(url, internal_links)
    
    # Batch processing: Split the internal_links into batches of 180
    batch_size = 180
    all_filtered_links = []
    # Process in batches
    for i in range(0, len(internal_links), batch_size):
        batch = internal_links[i:i + batch_size]
        filtered_batch = await filter_links_gpt(batch, file_name)
        all_filtered_links.extend(filtered_batch)
    
    filtered_links = list(set(all_filtered_links))

    
    new_links = []
    for link in filtered_links:
        # Check if we've processed or are pending on this URL
        if link not in processed_urls :
            processed_urls.add(link)
            new_links.append((link, depth + 1, file_name))
    
    # Add all new links to queue at once
    for link_info in new_links:
        await queue.put(link_info)
    







