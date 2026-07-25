import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
let idCounter = 0;
const nextId = () => ++idCounter;

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [books, setBooks] = useState([]);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  const fetchBooks = async () => {
    try {
      const res = await fetch(`${API_URL}/books`);
      const data = await res.json();
      setBooks(data.books || []);
    } catch (err) {
      console.error("Failed to fetch books", err);
    }
  };

  useEffect(() => {
    fetchBooks();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Types out text by updating the message with this exact id — 
  // never relies on array index, so it can't collide with other updates
  const typeOutMessage = (id, fullText, sources) => {
    let i = 0;
    const interval = setInterval(() => {
      i += 3;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? { ...m, text: fullText.slice(0, i), typing: i < fullText.length }
            : m
        )
      );

      if (i >= fullText.length) {
        clearInterval(interval);
        setMessages((prev) =>
          prev.map((m) => (m.id === id ? { ...m, text: fullText, sources, typing: false } : m))
        );
      }
    }, 15);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { id: nextId(), role: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.text }),
      });
      const data = await res.json();
      setLoading(false);

      const assistantId = nextId();
      setMessages((prev) => [
        ...prev,
        { id: assistantId, role: "assistant", text: "", sources: [], typing: true },
      ]);
      typeOutMessage(assistantId, data.answer, data.sources);
    } catch (err) {
      setLoading(false);
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "assistant", text: "Error: couldn't reach the server.", sources: [] },
      ]);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "system",
          text: res.ok
            ? `Uploaded "${data.book_title}" — ${data.chunks_created} chunks indexed.`
            : `Upload failed: ${data.detail || "unknown error"}`,
        },
      ]);
      if (res.ok) fetchBooks();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { id: nextId(), role: "system", text: "Upload failed: couldn't reach the server." },
      ]);
    } finally {
      setUploading(false);
      fileInputRef.current.value = "";
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">Library</div>
        <div className="book-list">
          {books.length === 0 && <div className="book-empty">No books indexed yet</div>}
          {books.map((book, i) => (
            <div key={i} className="book-card">
              <div className="book-title">{book.title}</div>
              <div className="book-stats">
                {book.page_count} pages · {book.chunk_count} chunks
              </div>
            </div>
          ))}
        </div>
        <button
          className="sidebar-upload-btn"
          onClick={() => fileInputRef.current.click()}
          disabled={uploading}
        >
          {uploading ? "Indexing..." : "+ Add Book"}
        </button>
        <input
          type="file"
          accept="application/pdf"
          ref={fileInputRef}
          onChange={handleUpload}
          style={{ display: "none" }}
        />
      </aside>

      <main className="main">
        <header className="header">
          <h1>BookPilot</h1>
          <p>Agentic RAG over your textbooks, with page-level citations.</p>
        </header>

        <div className="chat-window">
          {messages.length === 0 && (
            <div className="empty-state">Upload a book or ask a question to begin.</div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role} fade-in`}>
              {msg.role === "system" ? (
                <div className="system-msg">{msg.text}</div>
              ) : (
                <>
                  <div className="bubble">
                    {msg.text}
                    {msg.typing && <span className="cursor">|</span>}
                  </div>
                  {msg.sources && msg.sources.length > 0 && !msg.typing && (
                    <div className="sources fade-in">
                      <strong>Sources</strong>
                      <ul>
                        {msg.sources.map((s, idx) => (
                          <li key={idx}>{s}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              )}
            </div>
          ))}

          {loading && (
            <div className="message assistant fade-in">
              <div className="bubble typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="input-bar">
          <button
            className="upload-btn"
            onClick={() => fileInputRef.current.click()}
            disabled={uploading}
            title="Upload a PDF"
          >
            +
          </button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask something about your books..."
            rows={1}
          />

          <button className="send-btn" onClick={sendMessage} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;