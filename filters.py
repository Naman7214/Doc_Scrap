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
-Also filter out repeating URLs, the output should not contain any repeating URLs.
-You can assume that the URLs are valid and well-formed.
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
    "https://ai.google.dev/gemini-api/docs/migrate#search_grounding",
    "https://docs.pinecone.io/reference/api/introduction",
    "https://docs.pinecone.io/reference/api/introduction",
    "https://docs.pinecone.io/",

    
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


summary_links_prompt = """
Prompt:
"I have a list of URLs, and I need to filter the top 7 that provide substantial knowledge about the website and page content. The selected links should be those that contain valuable, structured, or informative content that can be used to generate a brief summary of the website and the specific page.

The filtering criteria should prioritize:

- **Informational Depth** – The page provides detailed insights, explanations, or structured knowledge rather than just navigation or promotional content.
- **Context Relevance** – The content contributes to understanding what the website is about and its purpose.
- **Unique Value** – It is not redundant compared to other links but adds a distinct perspective.
- **Minimal Noise** – Avoids excessive ads, login restrictions, or irrelevant details.

Return only the top 4 URLs that best meet these criteria in a structured JSON list format like this:


 [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4"
    ]

##NOTE##:
The Selected links should be capable to provide enough information to generate following data:
{
	"source_urls": [],
	"sdk_framework": "",
	"category": "",
	"chunk_id": "",
	"supported_languages": [],
	"versions": []
}
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

"""

summary_prompt = """
You are a text-processing AI that generates structured summaries from scraped technical documentation while preserving key information. The input consists of raw text from a single SDK or framework documentation. Your task is to create **a concise, meaningful summary** that captures essential details about the SDK, framework, or technology described in the source content.

### Summary Guidelines:
- **Concise & Informative**: The summary should be **brief yet detailed**, covering key features, functionality, and purpose.
- **Preserve Key Concepts**: Retain important **technical details, capabilities, and use cases**.
- **Avoid Redundancy**: Ensure that the summary adds value without unnecessary repetition.

### Metadata Extraction:
Since the entire request belongs to the **same SDK or framework**, extract and include the following metadata **once per request**:

- **"source_urls"**: A list of URLs from which the content was scraped.
- **"sdk_framework"**: Specifies whether the document is about an **SDK** or a **Framework**.
- **"category"**: The domain of the SDK or framework, selected from:
  - **AI**
  - **Cloud**
  - **Web**
  - **Mobile**
  - **Database**
  - **Security**
  - **DevOps**
- **"supported_languages"**: List of programming languages supported by the SDK/framework (e.g., Python, JavaScript, Java). If not specified, set as an empty list.
- **"versions"**: List of all versions mentioned in the documentation. If no version is found, set as an empty list.

### **Expected Output Format (JSON Object for One SDK/Framework)**:

```json
[
  {
    "chunked_data": "Gemini API provides powerful AI capabilities for text and image processing. It allows developers to integrate generative AI into applications with pre-trained models. The API supports multimodal inputs and has endpoints for text completion, image recognition, and structured data extraction. Designed for high performance and scalability, it is suitable for production-ready AI applications.",
    "metadata": {
      "source_urls": [
        "https://ai.google.dev/gemini-api/docs"
      ],
      "sdk_framework": "SDK",
      "category": "AI",
      "supported_languages": ["Python", "JavaScript"],
      "versions": ["1.2.0", "1.3.0"]
    }
  }
]
"""