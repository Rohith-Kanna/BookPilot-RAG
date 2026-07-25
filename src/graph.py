import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

load_dotenv()

# ---- Setup: load once, reuse across all queries ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
CHROMA_DIR = os.path.abspath(CHROMA_DIR)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# ---- State: the "memory" that flows through the graph ----
class AgentState(TypedDict):
    question: str
    needs_retrieval: bool
    retrieved_docs: List
    answer: str
    sources: List[str]


# ---- Node 1: Decide whether to retrieve ----
def decide_retrieval(state: AgentState) -> AgentState:
    question = state["question"]

    # llm for decision making wheter it needs RAG or not
    decision_prompt = f"""You are deciding if a question requires looking up textbook content, 
or if it's a general/greeting question you can answer directly.

Question: "{question}"

Reply with ONLY one word: "RETRIEVE" if this needs textbook lookup (specific facts, definitions, 
explanations of concepts likely in a Computer Networks or Operating Systems textbook), 
or "DIRECT" if it's a greeting, general chit-chat, or something you can answer without a textbook.
"""
    response = llm.invoke(decision_prompt)
    decision = response.content.strip().upper()

    state["needs_retrieval"] = "RETRIEVE" in decision
    print(f"[decide] '{question}' -> {'RETRIEVE' if state['needs_retrieval'] else 'DIRECT'}")
    return state


# ---- Node 2: Retrieve relevant chunks ----
def retrieve_documents(state: AgentState) -> AgentState:
    question = state["question"]
    docs = vectorstore.similarity_search(question, k=4)  # top 4 most relevant chunks

    state["retrieved_docs"] = docs
    state["sources"] = [
        f"{doc.metadata.get('book_title', 'Unknown')} - Page {doc.metadata.get('page', '?')}"
        for doc in docs
    ]
    print(f"[retrieve] Found {len(docs)} chunks")
    return state


# ---- Node 3: Generate the final answer ----
def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]

    if state.get("needs_retrieval") and state.get("retrieved_docs"):
        context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])
        prompt = f"""Answer the question using ONLY the context below. 
If the context doesn't contain the answer, say "I couldn't find this in the provided textbooks."

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"Answer this question directly: {question}"
        state["sources"] = []  # no sources for direct answers

    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


# ---- Routing function: tells LangGraph which path to take ----
def should_retrieve(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate"


# ---- Build the graph ----
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("decide", decide_retrieval)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    workflow.set_entry_point("decide")

    workflow.add_conditional_edges(
        "decide",
        should_retrieve,
        {"retrieve": "retrieve", "generate": "generate"},
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ---- Quick test ----
if __name__ == "__main__":
    app = build_graph()

    test_questions = [
        "Hi, how are you?",
        "What’s New in the Seventh Edition CN book?",
        "What is unique about this textbook?",
        "What topics does this Operating Systems book cover?",
        "Who is the author of this CN and OS book?"
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"Q: {q}")
        result = app.invoke({"question": q})
        print(f"A: {result['answer']}")
        print(f"Sources: {result.get('sources', [])}")