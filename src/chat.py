import os
from dotenv import load_dotenv
from src.search import vector_search

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """
Você é um assistente especializado que responde perguntas baseando-se estritamente em um banco de dados vetorial de documentos.
Você tem acesso às seguintes ferramentas:

{tools}

Para usar as ferramentas, siga ESTRITAMENTE o formato abaixo. Não adicione nenhum texto explicativo fora desse formato:

Thought: O que eu preciso fazer ou o que eu preciso buscar.
Action: a ferramenta a ser usada (uma das seguintes: {tool_names}).
Action Input: o texto da busca.
Observation: O resultado retornado pela ferramenta.
... (este ciclo Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: Eu tenho as informações necessárias para responder (ou não tenho).
Final Answer: [sua resposta final para o usuário].

REGRAS OBRIGATÓRIAS DE CONTEÚDO E FORMATO:
- Toda e qualquer resposta gerada por você deve obrigatoriamente iniciar com "Thought:". Nunca escreva nada antes de "Thought:".
- Responda somente com base no conteúdo retornado pelas ferramentas (Observation).
- Se a informação não estiver explicitamente no conteúdo retornado pelas ferramentas, sua resposta FINAL deve ser EXATAMENTE:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito nos documentos fornecidos.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Thought: A pergunta é sobre a capital da França, que está fora do escopo dos documentos disponíveis sobre IA.
Final Answer: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Thought: A pergunta busca dados sobre clientes em 2024, o que não consta nos documentos de IA.
Final Answer: "Não tenho informações necessárias para responder sua pergunta."

Comece!

Question: {input}
Thought:{agent_scratchpad}"""

def main():
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0
    )
    tools = [vector_search]

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    print("Bem-vindo ao chat! Digite 'sair' para encerrar.")
    while True:
        try:
            pergunta = input("Você: ").strip()
            if pergunta.lower() == 'sair':
                print("Encerrando o chat. Até logo!")
                break

            if not pergunta:
                print("Por favor, insira uma pergunta válida.")
                continue

            resultado = agent_executor.invoke({"input": pergunta})
            resposta = resultado.get("output", "Não tenho informações necessárias para responder sua pergunta.")
            print(f"Assistente: {resposta}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat. Até logo!")
            break
        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            continue

if __name__ == "__main__":
    main()