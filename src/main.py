import os
import shutil #used for file operations such as moving and deleting files
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .graph import build_graph
from .ingest import(
    load_and_chunk_pdf,
    create_new_vectorstore, 
    add_to_existing_vectorstore,
    vectorstore_exists,
    DATA_DIR
)


app = FastAPI(title="BookPilot API")

# Allows a frontend (running on a different port) to call this API later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev, tighten later if you deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the LangGraph agent once when the server starts (not per-request — reuse it)
agent_graph = build_graph()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    used_retrieval: bool

class UploadResponse(BaseModel):
    filename: str
    book_title: str
    pages_processed: int
    chunks_created: int
    status: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent_graph.invoke({"question": request.question})
    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        used_retrieval=result.get("needs_retrieval", False),
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to data/
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, file.filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Use filename (without .pdf) as the book title
    book_title = os.path.splitext(file.filename)[0]

    # Chunk it using the same logic as ingest.py
    chunks = load_and_chunk_pdf(filepath, book_title)

    # Add to store — create fresh if none exists yet, else append
    if vectorstore_exists():
        add_to_existing_vectorstore(chunks)
    else:
        create_new_vectorstore(chunks)

    # IMPORTANT: rebuild the graph so it picks up the updated vector store
    global agent_graph
    agent_graph = build_graph()

    return UploadResponse(
        filename=file.filename,
        book_title=book_title,
        pages_processed=len(set(c.metadata.get("page") for c in chunks)),
        chunks_created=len(chunks),
        status="success",
    )


@app.get("/books")
def list_books():
    """
    Returns all unique books in the vector store with their chunk 
    and page counts, derived from stored metadata.
    """
    from .graph import vectorstore

    # Chroma's underlying collection lets us fetch all metadata directly
    collection = vectorstore._collection
    all_data = collection.get(include=["metadatas"])

    books = {}
    for meta in all_data["metadatas"]:
        title = meta.get("book_title", "Unknown")
        page = meta.get("page", None)

        if title not in books:
            books[title] = {"chunk_count": 0, "pages": set()}

        books[title]["chunk_count"] += 1
        if page is not None:
            books[title]["pages"].add(page)

    result = [
        {
            "title": title,
            "chunk_count": data["chunk_count"],
            "page_count": len(data["pages"]),
        }
        for title, data in books.items()
    ]

    return {"books": result}

@app.get("/")
def health_check():
    return {"status": "BookPilot API is running"}