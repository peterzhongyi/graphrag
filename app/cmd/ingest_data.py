import os
import sys
import pathlib
import logging
from pathlib import Path
from google.cloud.storage import Client
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext, KnowledgeGraphIndex

# Add rag_demo package to PYTHONPATH
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from rag_demo import custom_schema, getenv_or_exit 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
NEO4J_URI = getenv_or_exit("NEO4J_URI")
NEO4J_USERNAME = getenv_or_exit("NEO4J_USERNAME")
NEO4J_PASSWORD = getenv_or_exit("NEO4J_PASSWORD")
INPUT_DIR = getenv_or_exit("INPUT_DIR")
OLLAMA_BASE_URL = os.getenv("OLLAMA_SERVER_URL", "http://ollama-service:11434")

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to {destination_file_name}")

def main():
    # Set up LlamaIndex
    Settings.llm = Ollama(
        model="gemma2:9b", 
        request_timeout=120.0,
        base_url=OLLAMA_BASE_URL
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # Download data from GCS
    bucket_name = "graphrag-eqwrnds"
    source_blob_name = "datalake/paul_graham_essay.txt"
    destination_file_name = "/tmp/data/paul_graham_essay.txt"
    download_from_gcs(bucket_name, source_blob_name, destination_file_name)

    # Load documents
    reader = SimpleDirectoryReader(input_dir=INPUT_DIR)
    documents = reader.load_data()
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create graph store
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

    # Create knowledge graph index
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        include_embeddings=True,
    )
    kg_index.storage_context.persist()
    logger.info("Knowledge graph index created and persisted")

if __name__ == "__main__":
    main()
