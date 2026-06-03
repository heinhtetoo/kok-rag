import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.constants import RECIPE_DIR, PARENT_STORE_PATH
from src.utils import save_parent

def ingest_recipe_chunks(filename: str = None) -> list[str]:
    chunks = []

    # Initialise a splitter that looks for natural breaks (newlines, periods)
    # Chunks are to be around 200 characters with a small overlap so context isn't lost
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", "."]
    )

    if filename in os.listdir(RECIPE_DIR) and filename.endswith(".txt"):
        file_path = os.path.join(RECIPE_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            title = filename.removesuffix(".txt")
            save_parent(parent_store_path=PARENT_STORE_PATH, parent_id=title, text=text)

            # Document splitting
            file_chunks = splitter.split_text(text)
            chunks.extend(file_chunks)

            print(f"[INFO] Loaded {filename}: Split into {len(file_chunks)} chunks.")

    return chunks