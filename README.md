# Project: Web Crawler, Chunking, and Embedding

This project provides a framework to crawl documentation sites, filter and chunk the content, generate embeddings, and then store those embeddings in Pinecone for retrieval and further analysis.

## Features

- **Crawling**: Uses Playwright-based async crawling to scrape pages up to a configurable depth.  
- **Filtering**: Leverages GPT-4o-mini to filter out irrelevant links.  
- **Chunking**: Breaks scraped content into semantically coherent pieces (uses GPT for chunk prompts).  
- **Embedding**: Generates both dense and sparse embeddings, suitable for hybrid search.  
- **Pinecone Integration**: Creates/manages an index and upserts chunk embeddings for retrieval.  

## Project Structure

- **main.py**: Orchestrates the entire flow — crawling, chunking, embedding, and Pinecone indexing.  
- **crawler.py**: Defines the crawling logic and link processing queue.  
- **chunker.py**: Splits raw scraped data into chunks and optionally generates summary chunks.  
- **embedding.py**: Adds embeddings to the chunked data.  
- **utils/**:  
  - `crawler_utils.py`, `chunking_utils.py`, `embedding_utils.py`, `pinecone_utils.py`: Helper functions for logging, code extraction, embedding, and Pinecone operations.  
- **filters.py**: Houses various prompts for filtering and chunking logic.  
- **testing/**: Contains scripts for testing and scoring retrieval.  
- **urls.py**: Entry points (URLs) for the crawling process.  
- **requirements.txt**: Dependency list (OpenAI, Pinecone, Crawl4AI, etc.).

## Setup

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**  
   - Copy your API keys into a `.env` file (e.g., OpenAI key, Pinecone key).  
   - Update `INDEX_NAME` or other parameters in `config.py` if needed.

3. **Run the Crawler**  
   ```bash
   python main.py
   ```

## Notes

- `crawl4ai` is employed for page navigation and text extraction.  
- GPT-driven selective filtering ensures only relevant pages are retained.  
- Chunking is done via GPT prompts to maintain logical content groupings.  
- Embeddings are generated with FastEmbed for dense vectors and BM25 for sparse vectors.

