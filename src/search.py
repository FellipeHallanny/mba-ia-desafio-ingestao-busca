import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import tool

load_dotenv()

def format_docs(docs):
    """Formata os documentos recuperados em uma string única"""
    return "\n\n".join([f"Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

@tool("vector_search")
def vector_search(query: str) -> str:
    """Busca informações no banco de dados vetorial para responder às perguntas do usuário."""
    # Verify environment variables
    for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")

    # Initialize embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_MODEL", "models/gemini-embedding-001")
    )

    # Initialize PGVector store
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )

    # Retrieve relevant documents
    docs = store.similarity_search(query, k=10)
    
    return format_docs(docs)
