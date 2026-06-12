# 👨‍🍳 Kök RAG — Your AI Sous-Chef

A production-grade **Retrieval-Augmented Generation (RAG)** system for culinary recipes, powered by local LLMs. Kök (Swedish for _cook_) lets you ingest recipes from the web, store them as vector embeddings, and ask natural-language questions — with answers grounded exclusively in your own recipe book.

---

## ✨ Features

- **Conversational Recipe Q&A** — Ask natural-language questions like _"How long do I simmer the beef?"_ and get precise, context-grounded answers.
- **Web Scraping Pipeline** — Ingest recipes directly from supported websites with a single API call.
- **Parent-Child Chunking** — Fine-grained retrieval with full-context reconstruction for higher answer quality.
- **Cross-Encoder Re-Ranking** — A dedicated re-ranking stage scores and filters retrieved documents for relevance before generation.
- **LLM-Driven Metadata Filtering** — Automatic extraction of cuisine and dish-type filters from user queries for targeted retrieval.
- **API-Key Authentication** — Secure all endpoints behind header-based API key validation.
- **Fully Containerised** — One-command deployment with Docker Compose; connects to a shared Ollama LLM service over a Docker network.
- **Interactive Web Dashboard** — A Streamlit chat UI for end-users, with source-chunk inspection.

---

## 🏗️ Architecture

```
┌─────────────┐        ┌──────────────────────────────────────────────────────┐
│  Streamlit  │───────▸│                   FastAPI Service                    │
│  Dashboard  │  HTTP  │                                                      │
│  (kok-ui)   │◂───────│  /ask ──▸ Filter Extraction ──▸ Vector Retrieval     │
└─────────────┘        │          ──▸ Parent Resolution ──▸ Cross-Encoder     │
                       │          ──▸ Prompt Augmentation ──▸ LLM Generation  │
                       │                                                      │
                       │  /ingest ──▸ Web Scraper ──▸ Chunking ──▸ Embeddin   │
                       └──────────────────────────┬───────────────────────────┘
                                                  │
                             ┌────────────────────┼────────────────────┐
                             │                    │                    │
                       ┌─────▼─────┐     ┌────────▼─────────┐   ┌──────▼──────┐
                       │ ChromaDB  │     │  Parent Store    │   │   Ollama    │
                       │ (Vectors) │     │  (JSON on disk)  │   │   (LLM)     │
                       └───────────┘     └──────────────────┘   └─────────────┘
```

### RAG Pipeline — `/ask`

| Stage                           | Description                                                                                                                                                    |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Filter Extraction**        | The LLM parses the user's query to extract structured metadata filters (`cuisine`, `dish_type`) — acting as a lightweight routing agent.                       |
| **2. Semantic Retrieval**       | ChromaDB performs a cosine-similarity search over child-chunk embeddings, optionally filtered by extracted metadata. Returns top 20 candidate chunks.          |
| **3. Parent Resolution**        | Retrieved child chunks are mapped back to their full parent documents via a JSON-based parent store, reconstructing complete recipe context.                   |
| **4. Cross-Encoder Re-Ranking** | A `cross-encoder/ms-marco-MiniLM-L-6-v2` model scores each parent document against the query. Only documents with a positive relevance score are kept (top 2). |
| **5. Prompt Augmentation**      | Surviving parent documents are injected into a system prompt with strict grounding instructions — the LLM must refuse to hallucinate.                          |
| **6. Generation**               | The Ollama-hosted LLM generates a final answer with `temperature=0.0` for deterministic output.                                                                |

### Ingestion Pipeline — `/ingest`

| Stage                        | Description                                                                                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Web Scraping**          | BeautifulSoup extracts structured recipe data (title, ingredients, instructions) from supported websites.                                                                        |
| **2. Parent-Child Chunking** | `RecursiveCharacterTextSplitter` from LangChain splits the full recipe into small child chunks (~150 chars, 20 char overlap). The full text is persisted as the parent document. |
| **3. Vector Embedding**      | Child chunks are embedded with `all-MiniLM-L6-v2` via Sentence Transformers and upserted into ChromaDB with rich metadata (`source`, `cuisine`, `dish_type`, `parent_id`).       |

---

## 🧰 Tech Stack

| Layer                | Technology                                                                                                       | Purpose                                                     |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **LLM Runtime**      | [Ollama](https://ollama.com/) + `qwen2.5:7b`                                                                     | Local inference — no API keys, no cloud dependency          |
| **API Framework**    | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)                                   | Async REST API with automatic OpenAPI docs                  |
| **Vector Database**  | [ChromaDB](https://www.trychroma.com/) (Persistent Client)                                                       | Cosine-similarity search over embeddings                    |
| **Embedding Model**  | [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2`                                             | Lightweight, high-quality sentence embeddings               |
| **Re-Ranker**        | [Cross-Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) — `ms-marco-MiniLM-L-6-v2`            | Pairwise relevance scoring for result refinement            |
| **Text Splitting**   | [LangChain Text Splitters](https://python.langchain.com/docs/how_to/recursive_text_splitter/)                    | Recursive character-based chunking with semantic separators |
| **Web Scraping**     | [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) + [Requests](https://docs.python-requests.org/) | Structured HTML parsing                                     |
| **Web Dashboard**    | [Streamlit](https://streamlit.io/)                                                                               | Chat-based UI with session state and source inspection      |
| **Data Validation**  | [Pydantic](https://docs.pydantic.dev/)                                                                           | Request/response schema enforcement                         |
| **Containerisation** | [Docker](https://www.docker.com/) + [Docker Compose](https://docs.docker.com/compose/)                           | Reproducible, multi-service deployment                      |
| **Language**         | Python 3.11                                                                                                      | Runtime                                                     |

---

## 📐 Architectural Considerations

### Parent-Child Chunking Strategy

Rather than retrieving small chunks directly for generation (which often strips away critical context), Kök implements a **two-level document hierarchy**:

- **Child chunks** (~150 characters) are embedded and used for high-precision semantic retrieval.
- **Parent documents** (full recipes) are stored in a JSON-based sidecar store and reconstructed at query time.

This ensures that the LLM always receives the **full recipe context** while retrieval remains granular and precise.

### Cross-Encoder Re-Ranking

A bi-encoder (Sentence Transformer) is fast but imprecise — it compares embeddings independently. The cross-encoder sees both the query and document together, producing significantly more accurate relevance scores. This two-stage retrieve-then-rerank pattern is an **industry-standard** approach used in production search systems at scale.

### LLM-as-Router for Metadata Filtering

Instead of requiring users to manually specify filters, the LLM acts as a **lightweight routing agent**, extracting structured metadata (`cuisine`, `dish_type`) from free-text queries. This enables filtered vector search without sacrificing the natural-language UX.

### Strict Grounding & Hallucination Prevention

- The system prompt explicitly instructs the LLM to refuse answering if context is insufficient.
- `temperature=0.0` ensures deterministic, non-creative output.
- Cross-encoder scoring with a `score > 0` threshold acts as a **relevance guardrail**, preventing low-quality context from reaching the LLM.

### Microservice Separation

The API server and the Streamlit dashboard run as **independent containers**, communicating over Docker's internal DNS. This enables:

- Independent scaling of the API and UI layers.
- Separation of concerns between the data/retrieval backend and the presentation layer.
- The ability to swap the UI entirely (e.g., with a mobile app) without touching the backend.

---

## 🏭 Industrial Standards

| Practice                      | Implementation                                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **API Security**              | Header-based API key authentication (`X-API-Key`) on all endpoints via FastAPI's dependency injection |
| **Input Validation**          | Pydantic models enforce typed request/response contracts with automatic 422 error responses           |
| **Environment Configuration** | Secrets and runtime config managed via `.env` files — never hardcoded                                 |
| **Containerisation**          | Slim Python base image, multi-service Compose, external network for LLM sharing                       |
| **Persistent Storage**        | Vector DB and data directories are volume-mounted, surviving container restarts                       |
| **Idempotent Writes**         | `upsert` operations prevent duplicate embeddings on re-ingestion                                      |
| **Error Handling**            | Structured HTTP error responses (`400`, `403`, `500`) with descriptive messages                       |
| **Separation of Concerns**    | Modular codebase — scraping, ingestion, embedding, chat, API, and UI in distinct modules              |
| **Version Control**           | Conventional commit messages (`feat:`, `fix:`, `chore:`, `refactor:`) for clear project history       |
| **Secrets Management**        | `.env` excluded from version control via `.gitignore`; vector DB artifacts also ignored               |

---

## 🚀 Getting Started

### Prerequisites

- **Docker** & **Docker Compose**
- **Ollama** running on the host or in a container, with the `qwen2.5:7b` model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```
- A shared Docker network for cross-container Ollama access:
  ```bash
  docker network create shared_llm_net
  ```

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/heinhtetoo/kok-rag.git
   cd kok-rag
   ```

2. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and set your KOK_API_KEY
   ```

3. **Start all services**

   ```bash
   docker compose up --build
   ```

4. **Access the application**
   | Service | URL |
   |---|---|
   | FastAPI (API + Docs) | [http://localhost:8000/docs](http://localhost:8000/docs) |
   | Streamlit Dashboard | [http://localhost:8501](http://localhost:8501) |

### Local Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the API server
uvicorn src.api:app --reload --port 8000

# Start the Streamlit UI (in a separate terminal)
streamlit run src/ui.py
```

---

## 📡 API Reference

### `POST /ask`

Ask a question about your recipes.

**Headers:** `X-API-Key: <your-api-key>`

```json
{
  "question": "What ingredients do I need for Mohinga?"
}
```

**Response:**

```json
{
  "answer": "For Mohinga, you will need...",
  "sources": ["Title: Mohinga\nSource: ...\n\nINGREDIENTS:\n..."]
}
```

### `POST /ingest`

Ingest a recipe from a supported URL.

**Headers:** `X-API-Key: <your-api-key>`

```json
{
  "url": "https://theburmalicious.com/recipe/mohinga",
  "cuisine": "Burmese",
  "dish_type": "Soup"
}
```

**Response:**

```json
{
  "message": "Successfully ingested with Parent-Child chunking!",
  "title": "mohinga.txt",
  "chunks_added": 12
}
```

---

## 📁 Project Structure

```
kok-rag/
├── src/
│   ├── api.py          # FastAPI application — /ask and /ingest endpoints
│   ├── chat.py         # Standalone CLI chat (development utility)
│   ├── constants.py    # Centralised path and collection name constants
│   ├── embed.py        # Vector embedding and ChromaDB upsert logic
│   ├── ingest.py       # Parent-child document chunking with LangChain
│   ├── scrape.py       # Web scraping for supported recipe sites
│   ├── ui.py           # Streamlit chat dashboard
│   └── utils.py        # LLM filter extraction, parent store I/O
├── data/recipes/       # Scraped recipe text files
├── vector_db/          # ChromaDB persistent storage (git-ignored)
├── Dockerfile          # Python 3.11-slim container image
├── docker-compose.yaml # Multi-service orchestration
├── requirements.txt    # Pinned Python dependencies
├── .env                # Environment variables (git-ignored)
└── .gitignore
```

---

## 📜 License

This project is for personal and educational use.
