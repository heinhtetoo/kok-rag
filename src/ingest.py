import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.constants import RECIPE_DIR

def ingest_recipe_chunks(filename: str = None) -> list[str]:
    chunks = []

    # Initialise a splitter that looks for natural breaks (newlines, periods)
    # Chunks are to be around 200 characters with a small overlap so context isn't lost
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", "."]
    )

    if filename in os.listdir(RECIPE_DIR) and filename.endswith(".txt"):
        file_path = os.path.join(RECIPE_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

            # Document splitting
            file_chunks = splitter.split_text(text)
            chunks.extend(file_chunks)

            print(f"Loaded {filename}: Split into {len(file_chunks)} chunks.")

    return chunks