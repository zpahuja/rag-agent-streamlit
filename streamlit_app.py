import aiohttp
import asyncio
import cohere
import json
import os
import requests
import streamlit as st
import tempfile
import time
import traceback

from concurrent.futures import ThreadPoolExecutor
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.readers.file import UnstructuredReader
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
from openai import OpenAI
from pathlib import Path
from playwright.async_api import async_playwright
from tenacity import retry, wait_random_exponential, stop_after_attempt

# CONSTANTS
OPENAI_MODEL = "gpt-4o"
QUERY_EXPANSION_FACTOR = 3
PLAYWRIGHT_TIMEOUT = 15000

# API KEYS
BING_API_KEY = st.secrets["BING_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
SCALE_SERP_API_KEY = st.secrets["SCALE_SERP_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

client = OpenAI()
co = cohere.Client(COHERE_API_KEY)


def fetch_news(query, topics=[], freshness="Week"):
    """Fetch news related to a query and optional topics using Bing News Search API, returning a list of news."""
    params_base = {
        "count": 100,  # Number of results to return
        "offset": 0,  # Offset for the results (for pagination)
        "freshness": freshness,  # Freshness of the news
        "safeSearch": "Moderate",  # Safe search filter
    }

    queries = [query] + [f"{query} {topic}" for topic in topics]
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    news_search_url = "https://api.bing.microsoft.com/v7.0/news/search"
    all_news = []

    async def fetch(session, q):
        params = params_base.copy()
        params["q"] = q
        async with session.get(news_search_url, params=params, headers=headers) as response:
            if response.status != 200:
                st.error(f"Failed to fetch news for query {q}: {response.status}")
                return []
            data = await response.json()
            return [
                {
                    "name": article.get("name", "No title provided"),
                    "url": article.get("url", "No link provided"),
                    "description": article.get("description", "No description provided"),
                    "datePublished": article.get("datePublished", "No date provided"),
                }
                for article in data.get("value", [])
            ]

    async def gather_news():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch(session, q) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if not isinstance(result, Exception):
                    all_news.extend(result)
                else:
                    st.error(f"Failed to process news response: {result}")

    asyncio.run(gather_news())
    return all_news


async def fetch_bing_search_results(query):
    """Fetch search results from Bing Search API asynchronously."""
    params = {
        "q": query,
        "count": 10,  # Number of results to return
        "offset": 0,  # Offset for the results (for pagination)
        "mkt": "en-US",  # Market code
        # "freshness": "Month"  # Date filter for results
    }
    headers = {
        "Ocp-Apim-Subscription-Key": BING_API_KEY,
    }
    search_url = "https://api.bing.microsoft.com/v7.0/search"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(search_url, params=params, headers=headers) as response:
                response.raise_for_status()  # Raises error for 4xx/5xx responses
                data = await response.json()
                return [
                    {
                        "title": result.get("name", "No title provided"),
                        "snippet": result.get("snippet", "No snippet provided"),
                        "link": result.get("url", "No link provided"),
                        "date": result.get("dateLastCrawled", "No date provided")
                    }
                    for result in data.get("webPages", {}).get("value", [])
                ]
        except aiohttp.ClientError as e:
            st.error(f"Failed to fetch Bing search results for query {query}: {e}")
            return []


async def fetch_scale_serp_results(query):
    """Fetch search results from Scale SERP API asynchronously."""
    params = {
        "api_key": SCALE_SERP_API_KEY,
        "q": query,
        "google_domain": "google.com",
        "include_answer_box": "false",
        "engine": "google",
    }
    search_url = "https://api.scaleserp.com/search"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(search_url, params=params) as response:
                response.raise_for_status()  # Raises error for 4xx/5xx responses
                data = await response.json()
                return [
                    {
                        "title": result.get("title", "No title provided"),
                        "snippet": result.get("snippet", "No snippet provided"),
                        "link": result.get("link", "No link provided"),
                    }
                    for result in data.get("organic_results", [])
                ]
        except aiohttp.ClientError as e:
            st.error(f"Failed to fetch Scale SERP results for query {query}: {e}")
            return []


def expand_query(user_query):
    system_prompt = f"""
        Create a JSON-formatted list of upto {QUERY_EXPANSION_FACTOR} Google search queries based on the user's query to find relevant articles.
        Follow this by crafting a concise ranker query, limited to 10 words, highlighting key keywords and concepts. 
        This ranker query will be used to prioritize the most relevant search results aggregated from the initial queries.

        The expected response structure is as follows:
        {{
            "queries": ["array of string queries"],
            "ranking_query": "a concise query to sort the amalgamated web results"
        }}
    """

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def google_search(query):
    # rewrite queries using LLM with json mode
    expanded_queries = expand_query(query)
    queries = expanded_queries.get("queries", [query])[:QUERY_EXPANSION_FACTOR]
    st.write({"action": "query expansion", "result": {"queries": queries}})

    # combine and deduplicate search results
    tik = time.time()

    async def gather_search_results(queries):
        tasks = [
            asyncio.create_task(fetch_bing_search_results(query)) for query in queries
        ]
        return await asyncio.gather(*tasks)

    search_results = asyncio.run(gather_search_results(queries))
    search_results = [result for sublist in search_results for result in sublist]
    unique_search_results = {
        result["link"]: result for result in search_results
    }.values()
    tok = time.time()
    st.write(f"Time to retrieve all SERP results: {tok - tik}")

    # scrapes search results in parallel
    urls = [result["link"] for result in unique_search_results]
    all_docs = []

    tik = time.time()

    async def scrape_and_save(url):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(
                    url, timeout=PLAYWRIGHT_TIMEOUT, wait_until="networkidle"
                )  # 5 seconds timeout
                page_source = await page.content()
                tmpdir = Path(tmpdirname)
                filepath = tmpdir / f"{url.replace('/', '_')}.html"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(page_source)
                loader = UnstructuredReader()
                docs = loader.load_data(file=filepath, split_documents=False)
                return docs
            except Exception as e:
                st.error(f"Timeout or error while trying to load {url}: {str(e)}")
                return []
            finally:
                await browser.close()

    async def gather_docs(urls):
        return await asyncio.gather(*[scrape_and_save(url) for url in urls])

    with tempfile.TemporaryDirectory() as tmpdirname:
        fetched_docs = asyncio.run(gather_docs(urls))
        for docs in fetched_docs:
            all_docs.extend(docs)

    tok = time.time()
    # st.write(f"Time taken to read and load all pages: {tok - tik}")

    # chunk, embed and store in vectore store with reference to source URL
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[4096, 1024, 256])
    nodes = node_parser.get_nodes_from_documents(all_docs)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    leaf_nodes = get_leaf_nodes(nodes)
    base_index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
    )

    base_retriever = base_index.as_retriever(similarity_top_k=20)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=False)

    rerank_query = expanded_queries.get("ranking_query", query)

    retrieved_results = retriever.retrieve(rerank_query)
    retrieved_docs = [doc.text for doc in retrieved_results]
    # st.write({
    #     "action": "retrieval",
    #     "query": rerank_query,
    #     "result": retrieved_docs
    # })

    # re-rank search results
    ranked_results = co.rerank(documents=retrieved_docs, query=rerank_query, top_n=7)
    ranked_docs = [
        {
            "text": retrieved_docs[result.index],
            "relevance_score": result.relevance_score,
        }
        for result in ranked_results.results
        if result.relevance_score >= 0.85
    ]
    # ranked_docs = [
    #     {
    #         "title": search_results[result.index]["title"],
    #         "snippet": search_results[result.index]["snippet"],
    #         "link": search_results[result.index]["link"],
    #         "relevance_score": result.relevance_score,
    #     }
    #     for result in ranked_results.results
    #     if result.relevance_score >= .85
    # ]
    # st.write({
    #     "action": "rerank",
    #     "result": {
    #         "query": rerank_query,
    #         "ranked documents": ranked_docs
    #     }
    # })

    return ranked_docs


def execute_tool_call(func_name, func_arguments):
    available_functions = {
        "google_search": google_search,
        "fetch_news": fetch_news,
    }
    if func_name in available_functions:
        func_to_call = available_functions[func_name]
        func_response = func_to_call(**func_arguments)
        st.write({"tool": func_name, "arguments": func_arguments, "response": func_response})
        return func_response
    else:
        st.error("Invalid tool call requested")


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(2))
def get_ai_response(user_message):
    messages = [{"role": "user", "content": user_message}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Executes web search for provided query and related queries and returns the most relevant snippets of information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant information on the web.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_news",
                "description": "Fetches news related to a query and optional topics within a specified date range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant news articles.",
                        },
                        "topics": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Optional list of topics to refine the news search.",
                        },
                        "freshness": {
                            "type": "string",
                            "enum": ["Day", "Week", "Month"],
                            "description": "Optional date range for the news results. Defaults to 'Week'.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]
    response_message = None
    try:
        while True:
            response = client.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, tools=tools
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            if tool_calls:
                messages.append(response_message)
                for tool_call in tool_calls:
                    function_response = execute_tool_call(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                    )
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": str(function_response),
                        }
                    )
            else:
                break
    except Exception as e:
        st.error("Unable to generate ChatCompletion response")
        st.error(f"Exception: {e}")
        st.error(f"Trace: {traceback.format_exc()}")
    return response_message.content


with st.form("search_form"):
    query = st.text_area("", placeholder="Ask me anything")
    submitted = st.form_submit_button("Search")

if submitted:
    if query:
        response = get_ai_response(query)
        st.write(response)
    else:
        st.write("Please enter a query.")