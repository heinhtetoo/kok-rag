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
from sentence_transformers import CrossEncoder

from src.scrape import scrape_recipe
from src.ingest import ingest_recipe_chunks
from src.embed import embed_chunks
from src.utils import extract_filters_from_query, load_parent_store
from src.constants import VECTOR_DB_DIR, COLLECTION_NAME, PARENT_STORE_PATH

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

# Initialize the Cross-Encoder for Re-Ranking
# This model is tiny (~90MB) and highly optimized for scoring relevance
print("Loading Cross-Encoder model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@app.post("/ask", response_model=QueryResponse, 
          dependencies=[Depends(verify_api_key)])
async def ask_kok(request: QueryRequest):
    try:
        # Extract filters from the user's query using the LLM
        extracted_filters = extract_filters_from_query(request.question, ollama_client, OLLAMA_MODEL)
        print(f"[INFO] Extracted Filters: {extracted_filters}")

        # Build metadata filter dynamically
        where_clause = {}
        conditions = []

        if extracted_filters.get("cuisine"):
            conditions.append({"cuisine": extracted_filters["cuisine"].capitalize()})

        if extracted_filters.get("dish_type"):
            conditions.append({"dish_type": extracted_filters["dish_type"].capitalize()})

        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # RETRIEVAL with conditions
        results = collection.query(
            query_texts=[request.question],
            n_results=20,
            where=where_clause if where_clause else None
        )

        if not results['documents'][0]:
            return QueryResponse(answer="I don't have that in your recipe book.",sources=[])
        
        # PARENT RESOLUTION: Find the unique parents of those 20 children
        children_metadata = results['metadatas'][0]
        unique_parent_ids = list(set([meta["parent_id"] for meta in children_metadata]))

        parent_store = load_parent_store(PARENT_STORE_PATH)
        candidate_parents = [parent_store[pid] for pid in unique_parent_ids if pid in parent_store]

        if not candidate_parents:
            return QueryResponse(answer="I couldn't reconstruct the recipe.", sources=[])
        
        # RE-RANKING: The Cross-Encoder scores the full recipes against the user's question
        # Create pairs: [(question, recipe1), (question, recipe2)]
        pairs = [[request.question, parent_text] for parent_text in candidate_parents]
        scores = cross_encoder.predict(pairs)

        # Combine the scores with the recipes and sort them from highest to lowest
        scored_parents = list(zip(scores, candidate_parents))
        scored_parents.sort(key=lambda x: x[0], reverse=True)

        # GUARDRAIL: Only take the top 1 or 2 best full recipes
        # Cross-Encoder scores are logits (often ranging from -10 to +10)
        top_parents = [doc for score, doc in scored_parents[:2] if score > 0]

        if not top_parents:
            return QueryResponse(answer="I found some recipes, but none seemed to directly answer your question.", sources=[])

        context = "\n\n---\n\n".join(top_parents)

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
            sources=top_parents # Returning the sources for UI debugging
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
        chunks, parent_id = ingest_recipe_chunks(recipe_file)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to ingest the recipe. No chunks were created.")

        # EMBED the chunks into the vector database
        chunks_added = embed_chunks(chunks, parent_id, collection, request.url, request.cuisine, request.dish_type)
        if not chunks_added:
            raise HTTPException(status_code=400, detail="Failed to embed the recipe chunks.")

        return IngestResponse(
            message="Successfully ingested with Parent-Child chunking!",
            title=recipe_file,
            chunks_added=chunks_added
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))