"""
RAG (Retrieval-Augmented Generation) Logic
-------------------------------------------
This file handles everything behind the scenes:
1. Loading a document (PDF or TXT)
2. Splitting it into small chunks
3. Turning chunks into embeddings (numbers that represent meaning)
4. Storing embeddings in ChromaDB (a vector database)
5. Answering questions by finding relevant chunks and asking the LLM
"""

import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

CHROMA_DIR = "./chroma_db"


def build_index(file_path: str):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


def ask(qa_chain, question: str) -> str:
    result = qa_chain.invoke({"query": question})
    return result["result"]