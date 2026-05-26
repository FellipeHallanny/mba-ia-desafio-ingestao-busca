import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()


def ingest_pdf():

    global k, loader, docs, chunks, embeddings, store, e

    for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
        if not os.getenv(k):
            raise RuntimeError(f"Environment variable {k} is not set")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    file_path = os.path.join(os.path.dirname(__file__), 'ia_intro.pdf')

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Arquivo `{file_path}` não encontrado")

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_MODEL", "models/gemini-embedding-001")
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )

    try:
        store.add_documents(documents=chunks)
    except Exception as e:
        print(f"Failed to add documents to PGVector: {e}")


if __name__ == "__main__":
    ingest_pdf()