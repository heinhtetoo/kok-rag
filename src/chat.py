import chromadb
from chromadb.utils import embedding_functions
import ollama

def ask_kok(query):
    # Connect to existing Vector Database
    client = chromadb.PersistentClient(path="vector_db")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="culinary_recipes", embedding_function=ef)

    # RETRIEVAL: Find the top 2 most relevant chunks
    results = collection.query(
        query_texts=[query], 
        n_results=2
    )

    # Combine the retrieved chunks into a single text block
    retrieved_chunks = results['documents'][0]
    context = "\n\n---\n\n".join(retrieved_chunks)

    # AUGMENTATION: Inject the context into the system prompt
    prompt = f"""
    You are a precise and helpful sous-chef, named Kök. Answer the user's question using ONLY the context provided below. 
    If the answer is not in the context, say "I don't have that in your recipe book." Do not make up cooking times or ingredients.

    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """

    print("\n[Thinking...]")

    # GENERATION: Send it to the local LLM
    response = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt
    )

    print("\n👨‍🍳 Kök:")
    print(response['response'])

if __name__ == "__main__":
    print("Welcome to your Local Culinary RAG! (Type 'exit' to quit)")
    while True:
        user_query = input("\nWhat would you like to know about your recipes? > ")
        if user_query.lower() == 'exit':
            break
        ask_kok(user_query)