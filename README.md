# Agente RAG Conversacional - Desafio de Ingestão e Busca (MBA em IA)

Este projeto foi desenvolvido como parte de um desafio prático de Engenharia de Software para o **MBA em IA**. Ele consiste em uma aplicação de **Retrieval-Augmented Generation (RAG)** em formato de Command Line Interface (CLI) stateless, utilizando **Python**, **LangChain**, **PostgreSQL com extensão pgvector** e a **API do Google Gemini**.

O objetivo principal é criar um agente conversacional inteligente que responde a perguntas baseando-se **estritamente** no conteúdo de um documento PDF fornecido (`src/ia_intro.pdf`), com forte blindagem contra alucinações e respostas fora de contexto.

---

## 🚀 Principais Funcionalidades

- **Ingestão Inteligente de Documentos**: Pipeline automatizado que carrega PDFs, realiza a quebra inteligente de texto em blocos (*chunking*) estruturados e gera embeddings semânticos.
- **Busca Semântica Avançada**: Utilização de busca por similaridade vetorial através da extensão `pgvector` do PostgreSQL para recuperar de forma ultra-rápida os trechos mais relevantes do documento.
- **Agente ReAct Conversacional**: Implementação de um agente de raciocínio e ação (*Reasoning and Acting*) usando LangChain, que interage com ferramentas de busca vetorial para embasar suas decisões.
- **Blindagem Total Contra Alucinações**: Prompt altamente estruturado que força o agente a usar exclusivamente o contexto recuperado. Perguntas fora do escopo ou que não constam no documento recebem uma resposta padronizada de ausência de informações.
- **Ambiente Conteinerizado**: Configuração pronta para o banco de dados via Docker e Docker Compose, simplificando o bootstrap do projeto.

---

## 🛠️ Stack Tecnológica

- **Linguagem**: [Python 3.10+](https://www.python.org/)
- **Framework de Orquestração**: [LangChain](https://www.langchain.com/) (Community, Core, Classic)
- **Provedor de LLM**: [Google Gemini API](https://ai.google.dev/) (Modelo `gemini-3.5-flash`)
- **Embeddings**: [Google Generative AI Embeddings](https://ai.google.dev/) (Modelo `models/gemini-embedding-001`)
- **Banco de Vetores**: [PostgreSQL](https://www.postgresql.org/) com a extensão [pgvector](https://github.com/pgvector/pgvector)
- **Infraestrutura Local**: [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
- **Processamento de PDF**: [PyPDF](https://pypdf.readthedocs.io/)

---

## 📋 Pré-requisitos

Antes de iniciar, certifique-se de ter os seguintes softwares instalados em sua máquina:

1. **Python 3.10 ou superior**
2. **Docker e Docker Compose**
3. **Chave de API do Google Gemini (Google AI Studio)**

---

## ⚙️ Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto copiando o arquivo de exemplo `.env.example` e configure os valores adequados:

```bash
# Linux/macOS
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```

Abaixo estão as variáveis presentes no arquivo:

| Variável | Tipo | Descrição | Exemplo / Padrão |
| :--- | :--- | :--- | :--- |
| `GOOGLE_API_KEY` | Obrigatória | Chave de API obtida no Google AI Studio | `AIzaSy...` |
| `PGVECTOR_URL` | Obrigatória | String de conexão para o banco de dados PostgreSQL | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PGVECTOR_COLLECTION` | Obrigatória | Nome da tabela/coleção que armazenará os embeddings | `pdf_collection` |
| `GEMINI_MODEL` | Opcional | Modelo utilizado para geração dos Embeddings | `models/gemini-embedding-001` |
| `CHUNK_SIZE` | Opcional | Quantidade máxima de caracteres em cada bloco de texto | `1000` |
| `CHUNK_OVERLAP` | Opcional | Sobreposição de caracteres entre blocos consecutivos | `150` |
| `OPENAI_API_KEY` | Opcional | Chave da OpenAI (caso decida alternar o provedor de LLM) | `sk-...` |
| `GOOGLE_MODEL` | Opcional | Modelo de chat alternativo do Google | `gemini-3.5-flash` |
| `OPENAI_MODEL` | Opcional | Modelo de chat alternativo da OpenAI | `gpt-4o-mini` |

---

## 🗺️ Arquitetura do Sistema

A aplicação adota um padrão clássico de RAG dividido em duas fases fundamentais: **Ingestão** (offline) e **Busca/Chat** (online).

### Estrutura de Diretórios

```text
mba-ia-desafio-ingestao-busca/
├── .env                  # Arquivo de configuração de chaves e parâmetros (não comitado)
├── .env.example          # Template de exemplo para as variáveis de ambiente
├── docker-compose.yml    # Definição do serviço PostgreSQL + pgvector
├── document.pdf          # PDF de referência original
├── requirements.txt      # Dependências e bibliotecas Python
├── src/
│   ├── __init__.py
│   ├── chat.py           # Interface de chat CLI e lógica do Agente ReAct
│   ├── ingest.py         # Pipeline de ingestão, quebra de texto e upload no pgvector
│   ├── search.py         # Implementação da ferramenta de busca vetorial
│   └── ia_intro.pdf      # PDF carregado pelo script de ingestão
└── venv/                 # Ambiente virtual Python (gerado localmente)
```

### Fluxo de Ingestão de Dados

```mermaid
graph TD
    A[src/ia_intro.pdf] -->|PyPDFLoader| B(Documento Completo Carregado)
    B -->|RecursiveCharacterTextSplitter| C(Chunks de Texto de 1000 caracteres)
    C -->|GoogleGenerativeAIEmbeddings| D(Vetores de Embeddings de 768 dimensões)
    D -->|PGVector Store| E[(PostgreSQL + pgvector)]
```

### Ciclo de Execução da Consulta (Agente ReAct)

```mermaid
sequenceDiagram
    actor Usuario as Usuário (Terminal CLI)
    participant Chat as src/chat.py (Agente ReAct)
    participant Search as src/search.py (Ferramenta Vector Search)
    participant DB as PostgreSQL (pgvector)
    participant Gemini as Google Gemini (gemini-3.5-flash)

    Usuario->>Chat: Envia pergunta ("Qual o conceito de IA?")
    Chat->>Gemini: Processa a entrada e define o Thought (Pensamento)
    Gemini-->>Chat: Solicita ação de busca (Action: vector_search)
    Chat->>Search: Executa vector_search(query)
    Search->>DB: Busca por similaridade cosseno (k=10 chunks)
    DB-->>Search: Retorna os chunks mais relevantes
    Search-->>Chat: Retorna texto consolidado (Observation)
    Chat->>Gemini: Fornece a pergunta + contexto da busca
    Gemini-->>Chat: Gera resposta estritamente ancorada (Final Answer)
    Chat->>Usuario: Exibe resposta para o usuário
```

---

## 🏃 Como Rodar a Aplicação localmente

Siga o passo a passo detalhado abaixo para configurar o ambiente do absoluto zero.

### 1. Clonar o Repositório e Navegar até a Pasta

```bash
git clone https://github.com/FellipeHallanny/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Configurar o Ambiente Virtual Python

Crie e ative um ambiente virtual para isolar as dependências do projeto:

```bash
# No Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# No Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

Instale todos os pacotes Python necessários:

```bash
pip install -r requirements.txt
```

### 4. Inicializar o Banco de Dados com Docker

Inicie o container do PostgreSQL já configurado com a extensão `pgvector`. O Docker Compose cuidará de inicializar o banco e executar o bootstrap para registrar a extensão de vetores.

```bash
docker compose up -d
```

> [!NOTE]
> Você pode verificar se o banco de dados está rodando perfeitamente e saudável através do comando `docker ps`.

### 5. Executar a Ingestão do PDF

Rode o script de ingestão. Ele lerá o arquivo `src/ia_intro.pdf`, aplicará o algoritmo de chunking e enviará os vetores ao PostgreSQL:

```bash
python -m src.ingest
```

### 6. Iniciar o Chat com o Agente

Com os dados indexados, agora você pode conversar com o agente no terminal. O agente usará a arquitetura ReAct para buscar no banco de dados e responder a você.

```bash
python -m src.chat
```

Para sair da conversa a qualquer momento, digite `sair` ou pressione `Ctrl + C`.

---

## 🤖 Regras de Negócio e Comportamento do Agente

Para garantir conformidade com o desafio acadêmico, o agente conversacional segue uma **política rígida de segurança e verdade**:

1. **Apenas Informação Oficial**: O agente só responde se a resposta estiver explicitamente contida nos documentos fornecidos pelo banco de dados de vetores.
2. **Tratamento de Perguntas Fora de Contexto**: Se você perguntar algo de conhecimento geral (ex: *"Qual é a capital da França?"* ou *"Quem descobriu o Brasil?"*) ou informações não contidas no PDF, o agente retornará **exatamente** a seguinte frase padrão:
   > **Não tenho informações necessárias para responder sua pergunta.**
3. **Imparcialidade e Literalidade**: O agente não cria deduções, opiniões ou conselhos adicionais que extrapolem o texto.

---

## 🛠️ Comandos e Scripts Úteis

| Comando | Descrição |
| :--- | :--- |
| `docker compose up -d` | Sobe o banco de dados PostgreSQL com `pgvector` em segundo plano |
| `docker compose down` | Desliga o banco de dados e remove os containers locais |
| `docker compose logs -f` | Acompanha os logs de transação e inicialização do banco de dados |
| `python -m src.ingest` | Executa o carregamento, fragmentação e vetorização do documento PDF |
| `python -m src.chat` | Inicializa o terminal interativo de perguntas e respostas com o Agente |

---

## 🛡️ Resolução de Problemas (Troubleshooting)

### 1. `Connection refused` ou erro de conexão com PostgreSQL
- **Causa**: O container do banco de dados Docker não está em execução ou a porta `5432` já está sendo usada por outra instância local do Postgres.
- **Solução**: Pare qualquer serviço Postgres local rodando em sua máquina e execute `docker compose down && docker compose up -d`.

### 2. `RuntimeError: Environment variable GOOGLE_API_KEY is not set`
- **Causa**: O arquivo `.env` não foi criado na raiz do projeto ou a chave não foi configurada corretamente.
- **Solução**: Certifique-se de que o arquivo está nomeado exatamente como `.env` na raiz do repositório e que a chave está no formato `GOOGLE_API_KEY=sua_chave_aqui`.

### 3. Erro de carregamento do PDF `ia_intro.pdf não encontrado`
- **Causa**: O script `ingest.py` procura pelo arquivo dentro de `src/ia_intro.pdf`.
- **Solução**: Verifique se o arquivo PDF de introdução à IA está localizado dentro da pasta `src/` com o nome exato de `ia_intro.pdf`.

### 4. `API key not valid` ao gerar embeddings ou chamar a LLM
- **Causa**: A chave de API do Gemini inserida no `.env` é inválida ou expirou.
- **Solução**: Acesse o [Google AI Studio](https://aistudio.google.com/), gere uma nova chave de API e atualize seu arquivo `.env`.

---

## 📝 Licença e Créditos

Este repositório foi desenvolvido com finalidade puramente acadêmica para o programa de **MBA em IA**.

Desenvolvido com 🧠 e 💻.
