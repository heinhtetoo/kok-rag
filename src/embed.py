import chromadb

def embed_chunks(chunks: list[str], collection: chromadb.Collection) -> int:
    # Chroma requires a unique ID for every single chunk
    ids = [f"recipe_chunk_{i}" for i in range(len(chunks))]

    # upsert to insert new, or update if already exists
    collection.upsert(
        documents=chunks,
        ids=ids
    )

    print(f"Successfully stored {len(chunks)} chunks in the vector database!")
    return len(chunks)