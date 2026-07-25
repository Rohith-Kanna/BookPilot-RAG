# 📘 BookPilot

**Agentic RAG system for querying textbooks and study PDFs - with exact page-level citations.**

BookPilot lets you upload PDFs (textbooks, notes, syllabi) and ask natural language questions about their content. Unlike simply uploading a document to a general-purpose chatbot, BookPilot is a purpose-built RAG pipeline that:

**Demo:**  ![BookPilot Demo](demo.png)


- **Cites the exact page and source document** for every answer - not just a vague reference
- **Intelligently decides whether to retrieve** from your documents or answer directly, using an LLM-based routing agent (built with LangGraph)
- **Refuses to hallucinate** - if the answer isn't in your uploaded documents, it says so instead of guessing
- **Supports multiple books simultaneously**, tagging which source each answer came from

---

## Why this isn't "just upload a PDF to ChatGPT"

| | Generic chatbot upload | BookPilot |
|---|---|---|
| Page-level citation | Rarely precise | Every answer cites exact page + book |
| Multi-document routing | Limited | Tags and distinguishes between multiple uploaded books |
| Cost at scale | Re-processes large context per session | Embeds once, retrieves only relevant chunks per query |
| Agentic decision-making | Not visible/controllable | Explicit decide → retrieve → generate graph (LangGraph) |
| Control over retrieval | None | Full control over chunking, embedding model, retrieval strategy |

This project demonstrates the ability to build the underlying RAG infrastructure that products like ChatGPT's document upload feature are built on top of - the actual skill companies hire AI engineers for.

---

## Architecture

```
                     ┌─────────────┐
   User uploads PDF ─▶   /upload   │──▶ Chunk (page-aware) ──▶ Embed ──▶ ChromaDB
                     └─────────────┘

                     ┌─────────────┐
    User asks Q  ────▶    /chat    │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │   Decide     │   (LLM judges: does this need retrieval?)
                     └──┬───────┬──┘
                        │       │
                 RETRIEVE     DIRECT
                        │       │
              ┌─────────▼──┐    │
              │  Retrieve   │    │
              │  (ChromaDB) │    │
              └─────────┬──┘    │
                        │       │
                     ┌──▼───────▼──┐
                     │   Generate   │  (Groq LLM - cites sources if retrieved)
                     └──────┬──────┘
                            │
                     Answer + Sources
```

**Flow:**
1. PDFs are loaded page-by-page (`PyPDFLoader`), preserving page numbers in metadata
2. Text is chunked (~1000 chars, 200 overlap) and tagged with book title + page number
3. Chunks are embedded (`sentence-transformers/all-MiniLM-L6-v2`) and stored in ChromaDB
4. On each query, a LangGraph agent first **decides** whether the question needs document retrieval or can be answered directly
5. If retrieval is needed, top-k relevant chunks are pulled and passed to the LLM (Groq, `llama-3.3-70b-versatile`) along with the question
6. The LLM answers using only the retrieved context, and the response is returned with the exact page/book sources attached

---

## Tech Stack

- **Backend:** FastAPI, Python
- **LLM:** Groq (`llama-3.3-70b-versatile`)
- **Agent orchestration:** LangGraph
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector store:** ChromaDB (local persistence)
- **Frontend:** React (Vite), custom UI (no component library)
- **PDF processing:** PyPDF / LangChain document loaders

---

## Features

- 📤 Upload any PDF via the UI - automatically chunked, embedded, and indexed
- 💬 Chat interface with typing animation and real-time responses
- 📚 Sidebar library view showing all indexed books with page/chunk counts
- 🎯 Page-level source citations on every retrieval-based answer
- 🤖 Agentic routing - general questions skip retrieval entirely, saving latency and cost
- 🚫 Anti-hallucination - explicitly states when an answer isn't found in the documents

---

## Setup & Installation

### Backend

```bash
git clone https://github.com/Rohith-Kanna/BookPilot-RAG.git
cd BookPilot-RAG

uv venv
.venv\Scripts\activate   # Windows
uv pip install -r requirements.txt

# Add your Groq API key
echo GROQ_API_KEY=your_key_here > .env

uvicorn src.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`. API docs available at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

### Ingest your own PDFs (optional, via script)

```bash
python src/ingest.py new    # creates a fresh vector store
python src/ingest.py add    # adds to an existing one
```

Or simply upload PDFs through the running app's UI.

---

## Known Limitations

- **Some PDFs with non-standard font encoding** (e.g., certain PowerPoint-exported notes) previously caused embedding failures due to invalid Unicode sequences - this has been fixed via UTF-8 sanitization during chunking, but PDFs with heavily corrupted text layers may still occasionally produce lower-quality chunks.
- **Local vector store persistence** - currently uses local ChromaDB storage rather than a hosted cloud vector database. This works well for local development and demos; a production deployment would use a hosted vector DB (e.g., Chroma Cloud, Qdrant, Pinecone) to support stateless/serverless hosting.
- **Not yet deployed to a public URL** - currently runs locally. Deployment to a platform supporting persistent storage (e.g., Fly.io, a VPS, or a hosted vector DB) is a planned next step.

---

## Planned Improvements

- [ ] Deploy backend + frontend to a public URL
- [ ] Migrate vector store to a hosted cloud provider
- [ ] Add conversation memory for follow-up questions
- [ ] Model selector (switch between Groq models)
- [ ] OCR fallback for scanned/image-only PDF pages
