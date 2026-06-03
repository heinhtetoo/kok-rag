import chromadb

def embed_chunks(chunks: list[str], parent_id: str, collection: chromadb.Collection, url: str, cuisine: str, dish_type: str) -> int:
    # Chroma requires a unique ID for every single chunk
    ids = [f"{parent_id}_child_{i}" for i in range(len(chunks))]

    # Generate metadata for each chunk
    metadatas = [{
        "parent_id": parent_id,
        "source": url,
        "cuisine": cuisine,
        "dish_type": dish_type
    } for _ in chunks]

    # upsert to insert new, or update if already exists
    collection.upsert(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    print(f"[INFO] Successfully stored {len(chunks)} chunks in the vector database!")
    return len(chunks)