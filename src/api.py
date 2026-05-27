import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import ollama
from ollama import Client

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
ollama_client = Client(host=OLLAMA_HOST)

# Initialise the FastAPI app
app = FastAPI(title="Kök RAG API")

# Define the data structure for incoming requests
class QueryRequest(BaseModel):
    question: str

# Define the data structure for the response
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# Global setup for the Vector DB to avoid reconnecting on every request
client = chromadb.PersistentClient(path="vector_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="culinary_recipes", embedding_function=ef)

@app.post("/ask", response_model=QueryResponse)
async def ask_kok(request: QueryRequest):
    try:
        # RETRIEVAL
        results = collection.query(
            query_texts=[request.question],
            n_results=2
        )

        retrieved_chunks = results['documents'][0]
        context = "\n\n---\n\n".join(retrieved_chunks)

        # AUGMENTATION
        prompt = f"""
        You are a precise and helpful sous-chef, named Kök. Answer the user's question using ONLY the context provided below. 
        If the answer is not in the context, say "I don't have that in your recipe book." Do not make up cooking times or ingredients.

        Context:
        {context}
        
        Question: {request.question}
        
        Answer:
        """

        # GENERATION
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt
        )

        return QueryResponse(
            answer=response['response'].strip(),
            sources=retrieved_chunks # Returning the sources for UI debugging
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))