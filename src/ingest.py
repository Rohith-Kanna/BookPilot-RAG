import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
CHROMA_DIR = os.path.abspath(CHROMA_DIR)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_and_chunk_pdf(filepath, book_title, max_pages=None):
    print(f"Loading {filepath}...")
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    if max_pages:
        pages = pages[:max_pages]
        print(f"  Limited to first {max_pages} pages for testing")

    print(f"  Loaded {len(pages)} pages")

    for page in pages:
        page.metadata["book_title"] = book_title

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # Filter out empty/None/whitespace-only chunks (common in notes PDFs with 
    # blank pages or image-only pages that have no extractable text)
    valid_chunks = []
    for c in chunks:
        if c.page_content and isinstance(c.page_content, str) and c.page_content.strip():
            valid_chunks.append(c)

    dropped = len(chunks) - len(valid_chunks)
    if dropped > 0:
        print(f"  Dropped {dropped} empty/invalid chunks")

    print(f"  Split into {len(valid_chunks)} usable chunks")
    return valid_chunks


def create_new_vectorstore(chunks):
    """Wipes and creates a fresh vector store from these chunks."""
    print(f"Creating NEW vector store with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("Done - new store created.")
    return vectorstore


def add_to_existing_vectorstore(chunks):
    """Adds new chunks to an already-existing persisted store, without touching existing data."""
    print(f"Adding {len(chunks)} new chunks to EXISTING vector store...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    vectorstore.add_documents(chunks)
    print("Done - existing store updated, old data untouched.")
    return vectorstore


def vectorstore_exists():
    """Check if a Chroma store already exists on disk."""
    return os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0

if __name__ == "__main__":
    # Usage:
    #   python src/ingest.py new      -> wipes and builds fresh (use for OS today)
    #   python src/ingest.py add      -> adds to existing store (use later for SQL pdf)
    mode = sys.argv[1] if len(sys.argv) > 1 else "new"

    if mode == "new":
        os_chunks = load_and_chunk_pdf(os.path.join(DATA_DIR, "OS.pdf"), "Operating Systems")
        create_new_vectorstore(os_chunks)

    elif mode == "add":
        sql_chunks = load_and_chunk_pdf(os.path.join(DATA_DIR, "sql.pdf"), "SQL - Unit II DBMS")
        add_to_existing_vectorstore(sql_chunks)

    else:
        print("Unknown mode. Use 'new' or 'add'.")