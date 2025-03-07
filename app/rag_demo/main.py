import os
import logging

from llama_index.core import Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import KnowledgeGraphIndex

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag_demo import custom_schema, getenv_or_exit 

logger = logging.getLogger()

MODEL_NAME = getenv_or_exit("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
NEO4J_URI = getenv_or_exit("NEO4J_URI")
NEO4J_USERNAME = getenv_or_exit("NEO4J_USERNAME")
NEO4J_PASSWORD = getenv_or_exit("NEO4J_PASSWORD")
OLLAMA_SERVER_URL = getenv_or_exit("OLLAMA_SERVER_URL")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    text: str

@app.on_event("startup")
def startup_db_client():
    # Set up LlamaIndex
    Settings.llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_SERVER_URL,
        request_timeout=120.0
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
    # Set up Neo4j connection and graph store
    graph_store = Neo4jGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database="neo4j"
    )
    
    # Create storage context with graph store
    storage_context = StorageContext.from_defaults(
        graph_store=graph_store
    )
    
    # Load knowledge graph index
    app.kg_index = KnowledgeGraphIndex.from_storage(storage_context)
    
    # Create query engine optimized for graph-based retrieval
    app.query_engine = app.kg_index.as_query_engine(
        response_mode="compact",
        include_text=True,
        embedding_mode="hybrid",  # Uses both graph structure and embeddings
        retriever_mode="keyword",  # Can be "keyword", "embedding", or "hybrid"
        max_paths=3,  # Number of paths to explore in the graph
    )

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(query: Query):
    try:
        response = app.query_engine.query(query.text)
        return {"response": str(response)}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": "Failed to process query", "details": str(e)}
