import chromadb
from chromadb.utils import embedding_functions
from ingest import load_and_chunk_recipes

def build_vector_db():
    print("Loading and chunking recipes...")
    chunks = load_and_chunk_recipes()

    print("\nInitialising ChromaDB Persistent Client...")
    # Save the database files directly to local folder
    client = chromadb.PersistentClient(path="vector_db")

    # Define local embedding model
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    print("Creating collection...")
    # get_or_create prevents errors while running the script multiple times
    collection = client.get_or_create_collection(
        name="culinary_recipes",
        embedding_function=ef
    )

    print("Embedding and storing chunks (Downloading the model on the first run takes a moment)...)")

    # Chroma requires a unique ID for every single chunk
    ids = [f"recipe_chunk_{i}" for i in range(len(chunks))]

    # upsert to insert new, or update if already exists
    collection.upsert(
        documents=chunks,
        ids=ids
    )

    print(f"Successfully stored {len(chunks)} chunks in the vector database!")

    # Test if the mathematical retrieval works
    print("\n--- Testing Retrieval ---")
    query = "How long should I simmer the beef?"
    print(f"Question: {query}")

    results = collection.query(
        query_texts=[query],
        n_results=2 # Get the top 2 most relevant chunks
    )

    print("\nTop Match Found:")
    print(results['documents'][0][0])

if __name__ == "__main__":
    build_vector_db()