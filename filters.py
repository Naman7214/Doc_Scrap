filter_prompt = """
###TASK###
You will be given a list of URLs that need to be scraped. However, some URLs (such as login, signup, support, external urls and similar non-relevant pages) should be excluded from the scraping process.
###CONTEXT###
The goal is to scrape web pages that contain documentation related to SDKs or frameworks for maintaining a vector database. Any pages that do not contribute meaningful information to this purpose should be excluded.
###INSTRUCTIONS###
-Retain URLs that contain relevant documentation or technical information about SDKs and frameworks for vector databases.
-Exclude URLs that are clearly unrelated, such as authentication pages (e.g., login, signup), support pages, or general account settings.
-Do not exclude any URLs unless you are 100% certain that they do not contribute to the task.
-Ensure that no critical links are mistakenly removed.
-Also Exclude urls that redirect to documentation in some different languages. I only want pages that are in English.
-Also Filter if there are repeating hyperlinks linking to same page.
-If the URL has # there is high probability that its a hyperlink.
###EXAMPLE###
INPUT:
[
    "https://docs.pinecone.io/",
    "https://status.pinecone.io",
    "https://app.pinecone.io/organizations/-/settings/support",
    "https://app.pinecone.io/?sessionType=login",
    "https://app.pinecone.io/?sessionType=signup",
    "https://docs.pinecone.io/guides/get-started/overview",
    "https://docs.pinecone.io/reference/api/introduction",
    "https://ai.google.dev/gemini-api/docs/migrate#json_response",
    "https://ai.google.dev/gemini-api/docs/migrate#search_grounding"
]
OUTPUT:
[
    "https://docs.pinecone.io/",
    "https://status.pinecone.io",
    "https://docs.pinecone.io/guides/get-started/overview",
    "https://docs.pinecone.io/reference/api/introduction",
    "https://ai.google.dev/gemini-api/docs/migrate"
]
"""

chunk_prompt = """
    You are a text-processing AI that chunks and structures scraped documentation data while preserving semantic meaning. The input consists of raw text from technical documentation. Your task is to split the text into meaningful chunks while extracting metadata for each chunk.

    ### Chunking Guidelines:
    - Maintain semantic meaning: Ensure each chunk contains a **complete concept, topic, or explanation**.
    - Preserve code blocks: If a chunk contains a **code snippet**, keep it within the same chunk.
    - Segment long sections logically: Chunk by **headings, subheadings, or topics** rather than splitting arbitrarily.

    ### Metadata Extraction:
    For each chunk, extract and include the following metadata:
    - "SDK/Framework_name": The **name** of the SDK or framework being described.
    - "source_url": The **original URL** from which the content was scraped (provided as input).
    - "sdk_framework": Binary classification:
      - **SDK** → If the document primarily discusses an SDK (e.g., Python SDK, Node.js SDK).
      - **Framework** → If the document primarily describes a development framework (e.g., TensorFlow, React, FastAPI).
    - "category": The **domain** the SDK or framework belongs to. Choose from the following:
      - **AI**
      - **Cloud**
      - **Web**
      - **Mobile**
      - **Database**
      - **Security**
      - **DevOps**
    - "has_code_snippet": True if the chunk contains a **code example**, otherwise False.
    - "version": The **version** of the SDK or framework, if mentioned in the text.
      - If **not explicitly available**, set it as null.

    ### Expected Output Format (JSON List of Chunks):
    
json
    [
  {
    "chunked_data": "Extracted meaningful text chunk...",
    "metadata": {
      "SDK/Framework_name": "Gemini API",
      "source_url": "https://ai.google.dev/gemini-api/docs",
      "sdk_framework": "SDK",
      "category": "AI",
      "has_code_snippet": true,
      "version": "1.2.0"
    }
  },
  {
    "chunked_data": "Another meaningful text chunk...",
    "metadata": {
      "SDK/Framework_name": "Gemini API",
      "source_url": "https://ai.google.dev/gemini-api/docs",
      "sdk_framework": "SDK",
      "category": "AI",
      "has_code_snippet": false,
      "version": null
    }
  }
]

###Additional Notes:###
- If the document contains multiple sections about different SDKs/Frameworks, ensure each section has its own relevant metadata.
- Ignore unnecessary elements like menus, navigation links, and unrelated footnotes.
- Code snippets should be preserved within their respective chunks.
- If a version number is not explicitly mentioned, set "version": null.
- The version should be of framework or sdk.
"""
