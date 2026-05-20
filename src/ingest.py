import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_recipes():
    recipe_dir = "data/recipes"
    chunks = []

    # Initialise a splitter that looks for natural breaks (newlines, periods)
    # Chunks are to be around 200 characters with a small overlap so context isn't lost
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""]
    )

    for filename in os.listdir(recipe_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(recipe_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Document splitting
            file_chunks = splitter.split_text(text)
            chunks.extend(file_chunks)

            print(f"Loaded {filename}: Split into {len(file_chunks)} chunks.")

    return chunks

if __name__ == "__main__":
    recipe_chunks = load_and_chunk_recipes()
    print(f"\nTotal chunks generated: {len(recipe_chunks)}")
    if recipe_chunks:
        print(f"Sample Chunk 1:\n---\n{recipe_chunks[0]}\n---")