import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from typing import Optional

from src.scrape import scrape_recipe
from src.ingest import ingest_recipe_chunks
from src.embed import embed_chunks
from src.constants import VECTOR_DB_DIR, COLLECTION_NAME

# Define the data structure for incoming requests
class QueryRequest(BaseModel):
    question: str
    cuisine_filter: Optional[str] = None
    dish_type_filter: Optional[str] = None

# Define the data structure for the response
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

class IngestRequest(BaseModel):
    url: str
    cuisine: str = "Unknown"        # e.g., "Burmese", "Italian"
    dish_type: str = "Unknown"      # e.g., "Soup", "Salad", "Main"

class IngestResponse(BaseModel):
    message: str
    title: str
    chunks_added: int

# Load environment variables from .env file
load_dotenv()

# Initialise the FastAPI app
app = FastAPI(title="Kök RAG API")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

MASTER_API_KEY = os.getenv("KOK_API_KEY")

if not MASTER_API_KEY:
    raise RuntimeError("KOK_API_KEY environment variable is missing!")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Dependency function to validate incoming API keys."""
    if api_key == MASTER_API_KEY:
        return api_key
    
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN,
        detail="Could not validate credentials. Invalid or missing API Key."
    )

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
ollama_client = Client(host=OLLAMA_HOST)

# Global setup for the Vector DB to avoid reconnecting on every request
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

@app.post("/ask", response_model=QueryResponse, 
          dependencies=[Depends(verify_api_key)])
async def ask_kok(request: QueryRequest):
    try:
        # Build metadata filter dynamically
        where_clause = {}
        conditions = []

        if request.cuisine_filter:
            conditions.append({"cuisine": request.cuisine_filter})
        if request.dish_type_filter:
            conditions.append({"dish_type": request.dish_type_filter})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # RETRIEVAL with conditions
        results = collection.query(
            query_texts=[request.question],
            n_results=5,
            where=where_clause if where_clause else None
        )

        retrieved_chunks = results['documents'][0]
        distances = results['distances'][0] if 'distances' in results else []

        # Check if the database is empty or if the results are irrelevant
        # Find closer matches in the ChromaDB
        valid_chunks = []
        for i, chunk in enumerate(retrieved_chunks):
            if i < len(distances) and distances[i] < 1.5:
                valid_chunks.append(chunk)

        # Stop if no valid chunks survived the threshold
        if not valid_chunks:
            return QueryResponse(
                answer="I don't have that in your recipe book.",
                sources=[]
            )

        context = "\n\n---\n\n".join(valid_chunks)

        # AUGMENTATION
        system_prompt = f"""
        You are a precise and helpful sous-chef, named Kök. Answer the user's question using ONLY the context provided below. 
        If the provided context does not contain the answer, politely say "I don't have that in your recipe book." 
        Do not make up cooking times, ingredients, or instructions. Format your response cleanly.

        Context:
        {context}
        """

        # GENERATION
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ],
            options={
                "temperature": 0.0,
                "top_p": 0.9
            }
        )

        return QueryResponse(
            answer=response['message']['content'].strip(),
            sources=retrieved_chunks # Returning the sources for UI debugging
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ingest", response_model=IngestResponse,
          dependencies=[Depends(verify_api_key)])
async def ingest_url(request: IngestRequest):
    try:
        # SCRAPE the recipe from the provided URL
        recipe_file = scrape_recipe(request.url)
        if not recipe_file:
            raise HTTPException(status_code=400, detail="Failed to scrape the provided URL. Please ensure it's a supported recipe URL.")
        
        # INGEST the scraped recipe into chunks
        chunks = ingest_recipe_chunks(recipe_file)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to ingest the recipe. No chunks were created.")

        # EMBED the chunks into the vector database
        chunks_added = embed_chunks(chunks, collection, request.url, request.cuisine, request.dish_type)
        if not chunks_added:
            raise HTTPException(status_code=400, detail="Failed to embed the recipe chunks.")

        return IngestResponse(
            message="Recipe ingested successfully.",
            title=recipe_file,
            chunks_added=chunks_added
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))