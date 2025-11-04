import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from src.search import PROMPT_TEMPLATE
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

load_dotenv()

for k in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GEMINI_MODEL", "models/gemini-embedding-001")
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

prompt = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template=PROMPT_TEMPLATE,
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def chat():
    print("Bem-vindo ao chat! Digite 'sair' para encerrar.")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == "sair":
            print("Encerrando o chat. Até mais!")
            break
        results = store.similarity_search_with_score(query=pergunta, k=10)
        for i, (doc, score) in enumerate(results, start=1):
            print("=" * 50)
            print(f"Resultado {i} (score: {score:.2f}):")
            print("=" * 50)

            print("\nTexto:\n")
            print(doc.page_content.strip())

            print("\nMetadados:\n")
            for k, v in doc.metadata.items():
                print(f"{k}: {v}")

if __name__ == "__main__":
    chat()