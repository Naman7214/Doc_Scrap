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
# Adding the selector hierarchy from the second code
SELECTOR_HIERARCHY = [
    "button[role='tab']",
    "div[role='tab']",
    "[class*='data-lang']",
    "[class*='language-']",
    "[role='option']",
    "select",
    "option",
    "button, div, span, li",
]

PROGRAMMING_LANGUAGES = {
    "http", "python", "javascript", "typescript", "rust", "java", "csharp", 
    "go", "curl", "json", "c#", "csharp", "node.js", "node", "npm", "yarn", "pnpm", "react", 
    "angular", "vue", "svelte", "sql", "php", "ruby", "twilio-cli","node","cpp",".net","stripe-cli","scala","r"
}

# Maximum number of concurrent browser contexts to use
MAX_CONTEXTS = 10
# Maximum number of tabs per browser context
MAX_TABS_PER_CONTEXT = 5
# Maximum concurrent click operations per page
MAX_CONCURRENT_CLICKS = 5

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

async def click_element_and_extract(page, element, text, seen_code_blocks):
    """Click an element and extract code after clicking."""
    snippets = []
    try:
        if not await element.is_visible():
            return [], text
            
        element_text = await element.inner_text(timeout=3000)
        element_text = element_text.strip().lower()

        if element_text in PROGRAMMING_LANGUAGES:
            print(f"Clicking: {element_text} in element")
            await element.click()
            await asyncio.sleep(0.5)  # Reduced sleep time
            
            # Extract code blocks after clicking
            code_blocks = await page.locator("pre code, pre, code, div[class*='bg-'] pre code, div[class*='bg-'] pre").all()
            for code_block in code_blocks:
                try:
                    code_text = await code_block.inner_text(timeout=3000)
                    code_text = code_text.strip()
                    if code_text and code_text not in seen_code_blocks:
                        seen_code_blocks.add(code_text)
                        snippets.append(code_text)
                except Error:
                    continue
            return snippets, element_text
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
    
    # Function to handle dropdown-based content extraction
    async def handle_dropdown_based_content(page):
        dropdown_elements = await page.locator("select").all()
        for dropdown in dropdown_elements:
            try:
                options = await dropdown.locator("option").all()
                for option in options:
                    option_text = await option.inner_text()
                    option_text = option_text.strip().lower()
                    if option_text in PROGRAMMING_LANGUAGES:
                        await dropdown.select_option(value=await option.get_attribute("value"))
                        await asyncio.sleep(0.5)  # Wait for content to load

                        # Extract code blocks after selecting the dropdown option
                        code_blocks = await page.locator("pre code, pre, code, div[class*='bg-'] pre code, div[class*='bg-'] pre").all()
                        for code_block in code_blocks:
                            try:
                                code_text = await code_block.inner_text(timeout=3000)
                                code_text = code_text.strip()
                                if code_text and code_text not in seen_code_blocks:
                                    seen_code_blocks.add(code_text)
                                    code_snippets.setdefault(option_text, []).append(code_text)
                            except Error:
                                continue
            except Exception as e:
                print(f"Skipping dropdown due to error: {e}")

    # Handle dropdown-based content
    await handle_dropdown_based_content(page)

    async def handle_tab_based_content(page):
        tab_elements = await page.locator("button[role='tab'], div[role='tab']").all()
        for tab in tab_elements:
            try:
                tab_text = await tab.inner_text()
                tab_text = tab_text.strip().lower()
                if tab_text in PROGRAMMING_LANGUAGES:
                    print(f"Clicking tab: {tab_text} from {url}")
                    await tab.click()
                    await asyncio.sleep(0.5)  # Wait for content to load

                    # Extract code blocks after clicking the tab
                    code_blocks = await page.locator("pre code, pre, code, div[class*='bg-'] pre code, div[class*='bg-'] pre").all()
                    for code_block in code_blocks:
                        try:
                            code_text = await code_block.inner_text(timeout=3000)
                            code_text = code_text.strip()
                            if code_text and code_text not in seen_code_blocks:
                                seen_code_blocks.add(code_text)
                                code_snippets.setdefault(tab_text, []).append(code_text)
                        except Error:
                            continue
            except Exception as e:
                print(f"Skipping tab due to error: {e}")

    # Handle tab-based content
    await handle_tab_based_content(page)


    # Step 1: Use improved selector hierarchy to find relevant elements
    for selector in SELECTOR_HIERARCHY:
        try:
            elements = await page.locator(selector).all()
            if not elements:
                continue
            
            filtered_elements = []
            for element in elements:
                if not await element.is_visible():
                    continue

                text = await element.inner_text(timeout=3000)
                text = text.strip().lower()

                if text in PROGRAMMING_LANGUAGES:
                    print(f"Found relevant element: {text} in {selector} on {url}")
                    filtered_elements.append(element)  
            # Process elements concurrently
            click_tasks = []
            for element in filtered_elements:
                click_tasks.append(click_element_and_extract(page, element, "", seen_code_blocks))
            
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




