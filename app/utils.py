import os
import sys
import json
import requests
from collections import OrderedDict
from typing import List, Dict
from langchain.pydantic_v1 import BaseModel
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_openai import AzureOpenAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv

# Load Env
library_path = Path(__file__).resolve().parents[0]
sys.path.append(str(library_path))
load_dotenv(str(library_path) + "/credentials.env")

embeddings = AzureOpenAIEmbeddings(deployment=os.environ["EMBEDDING_DEPLOYMENT_NAME"])

def get_search_results(
    query: str,
    indexes: List[str],
    k: int = 30,
    top_n: int = 5,
    reranker_threshold: float = 1.0,
    sas_token: str = ""
) -> OrderedDict:
    """
    Perform hybrid (semantic + vector) search across one or more Azure AI Search indexes.
 
    Args:
        query (str): The user query.
        indexes (List[str]): List of index names to search.
        k (int): Number of top results to return.
        reranker_threshold (float): Minimum reranker score to include result.
        sas_token (str): Optional SAS token for secure blob URLs.
 
    Returns:
        OrderedDict: Results sorted by descending reranker score.
    """
### adding query embeddungs for Hybrid Search
    query_embedding = embeddings.embed_query(query)

    headers = {"Content-Type": "application/json", "api-key": os.environ["AZURE_SEARCH_KEY"]}
    params = {"api-version": os.environ["AZURE_SEARCH_API_VERSION"]}
 
    agg_search_results = {}

    for index in indexes:
        search_payload = {
            "search": query,
            "select": "chunk_id, title, chunk",
            "searchFields": "title, chunk",
            "queryType": "semantic",
            "vectorQueries": [{"kind": "vector", "vector" : query_embedding, "fields": "text_vector", "k": k}],
            "semanticConfiguration" : "faqs-website-semantic-configuration",
            "captions": "extractive",
            "answers": "extractive",
            "count": "true",
            "top": top_n
            }
 
        response = requests.post(
            os.environ["AZURE_SEARCH_ENDPOINT"] + f"/indexes/{index}/docs/search",
            headers=headers,
            params=params,
            data=json.dumps(search_payload)
        )
 
        if response.status_code != 200:
            raise Exception(f"Search API error: {response.status_code} - {response.text}")
 
        agg_search_results[index] = response.json()
 
    content = {}
    for index, results in agg_search_results.items():
        for item in results.get("value", []):
            score = item.get("@search.rerankerScore", 0)
            if score >= reranker_threshold:
                chunk_id = item.get("chunk_id", "")
                content[chunk_id] = {
                    "chunk_id": chunk_id,
                    "title": item.get("title", ""),
                    "parent_id": item.get("parent_id", ""),
                    "content": item.get("chunk", ""),
                    "caption": item.get("@search.captions", [{}])[0].get("text", ""),
                    "score": score,
                    "index": index
                }
 
    # Order by score
    ordered_content = OrderedDict()
    for chunk_id in sorted(content, key=lambda x: content[x]["score"], reverse=True)[:k]:
        ordered_content[chunk_id] = content[chunk_id]
 
    return ordered_content
 
 
class CustomAzureSearchRetriever(BaseRetriever):
    """
    Custom LangChain retriever for Azure AI Search hybrid retrieval.
    """
 
    indexes: List[str]
    topK: int
    top_n: int
    reranker_threshold: float
 
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        ordered_results = get_search_results(
            query,
            indexes=self.indexes,
            k=self.topK,
            top_n = self.top_n,
            reranker_threshold=self.reranker_threshold
        )
 
        documents = []
        for result in ordered_results.values():
            documents.append(
                Document(
                    page_content=result["content"],
                    metadata={
                        "source": result["index"],
                        "chunk_id": result["chunk_id"],
                        "parent_id": result["parent_id"],
                        "title": result["title"],
                        "caption": result["caption"],
                        "score": result["score"]
                    }
                )
            )
        return documents