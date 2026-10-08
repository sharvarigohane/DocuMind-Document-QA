# DocQuery – RAG Document Assistant

Upload a PDF or text file and ask questions about it in plain language. DocQuery retrieves the most relevant passages with semantic search and uses Llama 3.1 to turn them into a clear answer.

<!-- Add 2 screenshots here (upload + indexed message, a question with its answer) -->
<!-- ![DocQuery answering a question](screenshots/demo.png) -->

## How It Works

```
PDF / TXT upload
      │
      ▼
Split into chunks (500 characters, 50 overlap)
      │
      ▼
Embed each chunk locally (all-MiniLM-L6-v2)
      │
      ▼
Store vectors in ChromaDB
      │
Question ──► retrieve the top 4 most similar chunks
                         │
                         ▼
          Llama 3.1 (via Groq) answers using those chunks
```

1. **Load:** `PyPDFLoader` reads PDFs and `TextLoader` reads text files.
2. **Chunk:** `RecursiveCharacterTextSplitter` cuts the text into 500-character pieces with a 50-character overlap, so sentences at chunk edges keep their context.
3. **Embed:** each chunk is converted to a vector with the `sentence-transformers/all-MiniLM-L6-v2` model, which runs on your machine. No embedding API is needed.
4. **Store:** vectors are saved in ChromaDB.
5. **Retrieve:** a question is embedded the same way, and the 4 most similar chunks are fetched.
6. **Answer:** the chunks and the question go to `llama-3.1-8b-instant` on Groq (temperature 0) through a LangChain `RetrievalQA` chain.

## Features

- Upload PDF or plain text files
- Semantic search, so questions match on meaning rather than exact keywords
- Local embeddings with no embedding API
- The index rebuilds only when a new file is uploaded
- Per-session state in Streamlit, so each browser tab has its own chain
- Errors during indexing are shown in the interface, and temporary files are always deleted

## Tech Stack

Python · Streamlit · LangChain · ChromaDB · Hugging Face Sentence-Transformers · Groq (Llama 3.1)

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/sharvarigohane/docquery-rag-assistant.git
cd docquery-rag-assistant
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Add your Groq API key

Get a free key at [console.groq.com](https://console.groq.com) and set it as an environment variable:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-key-here"

# macOS / Linux
export GROQ_API_KEY="your-key-here"
```

### 3. Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, upload a document, wait for the "indexed successfully" message, and ask a question.

The first run downloads the MiniLM embedding model, so the first upload takes longer.

## Project Structure

```
├── app.py             # Streamlit interface: upload, question box, answer
├── rag.py             # RAG logic: load, chunk, embed, store, retrieve, answer
└── requirements.txt
```

`chroma_db/` is created automatically when you index a document.

## Limitations

- **One document at a time.** Uploading a new file replaces the previous index.
- **Shared index folder.** The vector store lives in a single `chroma_db/` folder that is cleared on every upload, so two people using the same running instance would overwrite each other's index. It's built for single-user use.
- **No source display.** Answers don't show which passages they came from yet.
- **Text-based PDFs only.** Scanned PDFs with no selectable text won't produce usable chunks.
- **Model limits.** Llama 3.1 8B is a small model. It can still give incomplete or wrong answers, so check important answers against the document.

