import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model="embeddinggemma",
)

def load_documents(file_path: str) -> list[Document]:
    print('Loading PDF Documents...')

    loader = PyPDFDirectoryLoader(file_path)

    return loader.load()

def split_documents(documents: list[Document]) -> list[Document]:
    print('Splitting Documents...')

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    )

    return text_splitter.split_documents(documents)

def create_vectorstore(documents_chunks: list[Document]) -> None:
    print('Creating Vector Store...')

    vectorstore = FAISS.from_documents(
        documents=documents_chunks,
        embedding=embedding_model,
    )

    return vectorstore

def save_embeddings(vectorstore: FAISS, file_path: str) -> None:
    print('Saving Embeddings...')

    vectorstore.save_local(file_path)

if __name__ == "__main__":
    docs = load_documents('./data/')
    docs_chunks = split_documents(docs)
    vectorstore = create_vectorstore(docs_chunks)
    save_embeddings(vectorstore, './data/embeddings/')
