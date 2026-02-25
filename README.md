# 📚 Sistema de Indexação e Recuperação Semântica

## Publicações do Repositório IPEA

O projeto RagPub visa o desenvolvimento de um sistema de agentes assistentes inteligentes para o repositório de conhecimento do Ipea. A proposta central é criar uma solução robusta, baseada em IA, capaz de processar e interpretar todos os documentos do repositório, permitindo aos usuários interagir com o conteúdo por meio de um agente conversacional. Os principais recursos incluem:

* Recomendação personalizada de leituras;  
* Respostas a perguntas com base no contexto específico dos documentos (RAG);  
* Interação conversacional com fontes de dados (gráficos, imagens e tabelas) presentes nos documentos.

Para atingir esses objetivos, o projeto utilizará uma arquitetura de sistema multi-agentes, bibliotecas Python especializadas e técnicas avançadas de Recuperação Aumentada por Geração (RAG). O detalhamento do funcionamento interno será apresentado nas próximas seções.

Pipeline completo de ingestão, processamento e indexação vetorial híbrida (Dense + Sparse + ColBERT) para publicações 
institucionais do IPEA.

---

<!-- TOC -->
* [📚 Sistema de Indexação e Recuperação Semântica](#-sistema-de-indexação-e-recuperação-semântica)
  * [Publicações do Repositório IPEA](#publicações-do-repositório-ipea)
* [**1. Contexto**](#1-contexto-)
* [**2. Tecnologias**](#2-tecnologias-)
* [**3. Funcionamento**](#3-funcionamento-)
  * [**3.1. Ingestão**](#31-ingestão-)
* [🏗️ Arquitetura Geral](#-arquitetura-geral)
* [📂 Estrutura do Projeto](#-estrutura-do-projeto)
* [🗃️ Banco de Controle (SQLite)](#-banco-de-controle-sqlite)
    * [Status possíveis](#status-possíveis)
* [🌐 Scraper do Repositório](#-scraper-do-repositório)
* [🧠 Pipeline de Ingestão](#-pipeline-de-ingestão)
* [📥 Download e Cache de PDFs](#-download-e-cache-de-pdfs)
* [✂️ Chunking](#-chunking)
  * [Semantic Chunker (Principal)](#semantic-chunker-principal)
  * [Simple Chunker (Fallback)](#simple-chunker-fallback)
* [🔎 Indexação Vetorial (Qdrant)](#-indexação-vetorial-qdrant)
  * [Criação da coleção](#criação-da-coleção)
    * [Modelos utilizados](#modelos-utilizados)
  * [Criação de índices de payload](#criação-de-índices-de-payload)
  * [Teste de ingestão](#teste-de-ingestão)
* [📦 Estrutura do Payload](#-estrutura-do-payload)
* [🖼️ Extração de Imagens](#-extração-de-imagens)
* [🔐 Robustez Operacional](#-robustez-operacional)
* [🚀 Execução](#-execução)
  * [1️⃣ Configurar variáveis](#1-configurar-variáveis)
  * [2️⃣ Criar coleção](#2-criar-coleção)
  * [3️⃣ Criar índices](#3-criar-índices)
  * [4️⃣ Executar scraping](#4-executar-scraping)
  * [5️⃣ Executar ingestão](#5-executar-ingestão)
* [🔌 API](#-api)
<!-- TOC -->

---

# **1. Contexto**  
O projeto RagPub visa o desenvolvimento de um sistema de agentes assistentes inteligentes para o repositório de conhecimento do Ipea. A proposta central é criar uma solução robusta, baseada em IA, capaz de processar e interpretar todos os documentos do repositório, permitindo aos usuários interagir com o conteúdo por meio de um agente conversacional. Os principais recursos incluem:

* Recomendação personalizada de leituras;  
* Respostas a perguntas com base no contexto específico dos documentos (RAG);  
* Interação conversacional com fontes de dados (gráficos, imagens e tabelas) presentes nos documentos.

Para atingir esses objetivos, o projeto utilizará uma arquitetura de sistema multi-agentes, bibliotecas Python especializadas e técnicas avançadas de Recuperação Aumentada por Geração (RAG). O detalhamento do funcionamento interno será apresentado nas próximas seções.

# **2. Tecnologias**  
O projeto está sendo desenvolvido em Python, com código fonte disponível em [IpeaPub](https://github.com/ipeadata-lab/IpeaPub) no GitHub do Ipea. Os principais frameworks e bibliotecas utilizadas no projeto são:

* Docling (para parsing de documentos)  
* Qdrant (para banco vetorial)  
* PyTorch/transformers (para rodar modelos de IA localmente)  
* BeautifulSoup (para o crawler do site do Ipea)
* FastAPI (para receber e responder requisições)  
* Docker (Conteinerização do projeto)

# **3. Funcionamento**  
Existem 2 etapas para o funcionamento integral do sistema: ingestão e recuperação. Abaixo, será explicado cada uma delas. Os arquivos Python envolvidos em cada sub etapa do projeto estará explicitado entre parênteses para ajudar o entendimento.

## **3.1. Ingestão**  
Essa etapa serve para inserir as informações dos documentos do repositório na base de dados do projeto. As informações são a base para o funcionamento do sistema multiagente, por isso pode ser considerada a etapa mais importante do projeto. São utilizados bancos de dados vetoriais para realizar a busca semântica de informações.


1. (*scraper.py* e *banco\_metadados.py*) Todos os metadados de todos os documentos são adicionados em um banco de dados relacional âncora, com um estado de processamento pendente.  
2. (*docling\_pipeline.py* e *banco\_vetorial.py)* A pipeline busca por documentos pendentes no banco relacional e processa as informações dele para inserir nas coleções de pontos vetoriais: chunks, tabelas, imagens e recomendação, utilizando o Docling como ferramenta principal.

O arquivo *utils.py* possui funções auxiliares, como o crawler do site para buscar os arquivos PDFs a serem processados. Na pipeline do Docling, encontra-se também acesso a modelos de LLM e visão computacional para resumir e legendar tabelas e imagens, para que possam ser buscadas na query semântica do banco vetorial. Os arquivos de processamento se encontram na pasta *ingestor*, relacionado justamente com a etapa de ingestão.

---

# 🏗️ Arquitetura Geral

```
Scraper → SQLite (Banco 1) → Ingestão → Chunking → Embeddings
→ Qdrant (Banco Vetorial) → API de Consulta
```

O sistema é dividido em dois bancos:

| Camada  | Tecnologia | Finalidade                       |
| ------- | ---------- | -------------------------------- |
| Banco 1 | SQLite     | Controle operacional da ingestão |
| Banco 2 | Qdrant     | Armazenamento vetorial híbrido   |

---

# 📂 Estrutura do Projeto

```
.
├── create_collection.py
├── create_indexes.py
├── create_ingestion.py
├── teste_ingestion.py
├── run.py
├── scraper.py
│
├── ingestao/
│   ├── db/
│   │   └── banco_metadados.py
│   └── utils/
│       ├── clean_itens.py
│       ├── semantic_chunker.py
│       └── simple_chunker.py
│
├── cache/pdfs/
├── logs/
├── img/
└── data/banco1.db
```

---

# 🗃️ Banco de Controle (SQLite)

Arquivo: 

Tabela principal:

```sql
documentos (
    id TEXT PRIMARY KEY,
    titulo TEXT,
    autores TEXT,
    ano INTEGER,
    tipo_conteudo TEXT,
    resumo TEXT,
    palavras_chave TEXT,
    link_pdf TEXT,
    link_download TEXT,
    status_ingestao TEXT,
    data_ingestao TEXT
)
```

### Status possíveis

* `pendente`
* `em processamento`
* `sem_pdf`
* `processado`
* `erro`

A ingestão é incremental e resiliente a falhas.

---

# 🌐 Scraper do Repositório

Arquivo: 

Funções:

* Consome API do repositório IPEA
* Extrai metadados estruturados
* Normaliza campos
* Persiste no SQLite
* Remove duplicatas posteriormente

Controle de duplicidade baseado em:

```
titulo + ano + resumo
```

---

# 🧠 Pipeline de Ingestão

Arquivo: 

Fluxo completo:

1. Busca documento pendente no SQLite
2. Download real do PDF (com retry + cache SHA256)
3. Extração via Docling (OCR + imagens)
4. Limpeza textual avançada
5. Chunking semântico
6. Geração de embeddings híbridos
7. Upload batch para Qdrant
8. Atualização de status

---

# 📥 Download e Cache de PDFs

Arquivo: 

Características:

* Retry automático
* Verificação de assinatura `%PDF`
* Cache por hash SHA256
* Armazenamento em `cache/pdfs/`

---

# ✂️ Chunking

## Semantic Chunker (Principal)

Arquivo: 

Características:

* Embeddings SentenceTransformers
* Clusterização HDBSCAN
* Agrupamento por similaridade semântica
* Controle rígido de tokens (max_tokens=290)

## Simple Chunker (Fallback)

Arquivo: 

* Split por sentenças
* Controle direto de tokens
* Sem clusterização

---

# 🔎 Indexação Vetorial (Qdrant)

## Criação da coleção

Arquivo: 

Configuração híbrida:

```python
vectors_config = {
    "dense": 768 (cosine),
    "colbert": 128 (multi-vector MAX_SIM)
}

sparse_vectors_config = {
    "sparse": BM25
}
```

### Modelos utilizados

| Tipo    | Modelo                                                      |
| ------- | ----------------------------------------------------------- |
| Dense   | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| Sparse  | Qdrant/bm25                                                 |
| ColBERT | colbert-ir/colbertv2.0                                      |

---

## Criação de índices de payload

Arquivo: 

Cria índices KEYWORD para:

* metadata.ticker
* metadata.form_type
* metadata.source

---

## Teste de ingestão

Arquivo: 

Valida:

* Conexão com Qdrant
* Estrutura vetorial híbrida
* Upsert de ponto sintético

---

# 📦 Estrutura do Payload

Cada chunk gera:

```json
{
  "text": "...",
  "metadata": {
    "document_id": "...",
    "titulo": "...",
    "autores": "...",
    "ano": 2023,
    "tipo_conteudo": "...",
    "palavras_chave": "...",
    "pagina": 12,
    "imagens_pagina": [...]
  }
}
```

---

# 🖼️ Extração de Imagens

* Extraídas por página via Docling
* Salvas em `img/{pdf_hash}/`
* Caminhos armazenados no payload

---

# 🔐 Robustez Operacional

* Controle transacional de status
* Batch upload resiliente
* Log individual por documento em `logs/`
* Retry HTTP automático
* Proteção contra overflow de tokens no ColBERT
* Flush final garantido de buffer

---

# 🚀 Execução

## 1️⃣ Configurar variáveis

`.env`:

```
QDRANT_URL=http://localhost:6333
```

---

## 2️⃣ Criar coleção

```bash
python create_collection.py
```

---

## 3️⃣ Criar índices

```bash
python create_indexes.py
```

---

## 4️⃣ Executar scraping

Arquivo: 

```bash
python run.py
```

---

## 5️⃣ Executar ingestão

```bash
python create_ingestion.py
```

---

# 🔌 API

A camada de API já está implementada separadamente e consome:

* Coleção `publicacoes_ipea`
* Recuperação híbrida
* Payload estruturado

A API não depende do pipeline de ingestão em tempo real, apenas da coleção indexada.

