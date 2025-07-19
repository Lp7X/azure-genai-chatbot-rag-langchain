import os
import sys
from operator import itemgetter
from datetime import datetime
from typing import Any, Dict, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from langserve import add_routes
from langchain_openai import AzureChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField, ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import CosmosDBChatMessageHistory
from langchain.chains import RetrievalQA
from pathlib import Path
from dotenv import load_dotenv

from .utils import CustomAzureSearchRetriever
from .prompts import CUSTOM_CHATBOT_PROMPT

# Load Env
library_path = Path(__file__).resolve().parents[0]
sys.path.append(str(library_path))
load_dotenv(str(library_path) + "/credentials.env")

###################################
# Env variable needed by langchain
os.environ["AZURE_OPENAI_MODEL_NAME"] = os.environ["GPT4_DEPLOYMENT_NAME"]
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]

# Declaration of the App
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangChain app using RetrievalQA with Azure AI Search",
)


# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Default route -> leads to the OpenAPI Swagger definition
@app.get("/{path_name:path}")
async def catch_all_redirect(path_name: str):
    return RedirectResponse("/docs")

###################### Now a complex agent

# History function
def get_session_history(session_id: str, user_id: str) -> CosmosDBChatMessageHistory:
    cosmos = CosmosDBChatMessageHistory(
        cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
        cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
        cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
        connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
        session_id=session_id,
        user_id=user_id
        )
    # prepare the cosmosdb instance
    cosmos.prepare_cosmos()
    return cosmos


# Set LLM
llm = AzureChatOpenAI(deployment_name=os.environ.get("AZURE_OPENAI_MODEL_NAME"), temperature=0.5, max_tokens=1500, streaming=True)

# Initialize our Tools/Experts
doc_indexes = ["faqs-website"]

retriever = CustomAzureSearchRetriever(indexes=doc_indexes, topK=30, top_n=5, reranker_threshold=1.0)

qa_chain = (
    {
        "context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "history": itemgetter("history")
    }
    | CUSTOM_CHATBOT_PROMPT
    | llm
    | (lambda x: {"output": x.content})
)

qa_chain_with_memory = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

# Create Input and Output Schemas

class Input(TypedDict):
    question: str

class Output(BaseModel):
    output: str

# Add API route

add_routes(
    app,
    qa_chain_with_memory.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

###################### Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)