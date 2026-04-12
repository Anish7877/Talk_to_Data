# Talk to Data

A 6-layer AI-powered natural language to SQL/data pipeline system powered by **Groq API** for ultra-fast LLM inference.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    System Architecture                       │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Semantic Cache     → Skip LLMs on repeated queries │
│  Layer 2: Intent Router      → Classify: SQL vs RAG vs Both  │
│  Layer 3: TAG / Metadata     → Dynamic schema retrieval      │
│  Layer 4: Multi-Agent SQL    → Planner → Coder → Validator   │
│  Layer 5: Secure Execution   → Read-only database sandbox    │
│  Layer 6: Storyteller        → Natural language + audit trail│
└──────────────────────────────────────────────────────────────┘
```

## Features

- **Groq-Powered**: Ultra-fast inference with Llama 3.1 and Mixtral models
- **Semantic Caching**: Skip LLM calls for semantically similar queries
- **Intelligent Routing**: Automatically routes to SQL, RAG, or hybrid pipelines
- **Dynamic Schema**: Vector-based schema retrieval without hardcoding
- **Multi-Agent SQL**: Three specialized agents (planner, coder, validator)
- **Secure Execution**: Read-only database role with parameterized queries
- **Full Audit Trail**: Complete lineage tracing for every query

## Prerequisites

- Python 3.11+
- **Groq API key** (free at console.groq.com)
- PostgreSQL (local or Docker)
- ChromaDB (local, free)
- Redis (Docker)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Groq API key and database credentials
```

### 3. Start Infrastructure

```bash
# Start Redis
docker run -d -p 6379:6379 redis

# Start PostgreSQL (example)
docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=secret postgres
```

### 4. Set Up Database Role

```python
from layers import DatabaseRoleManager

# Run in psql:
for sql in DatabaseRoleManager.create_readonly_role_sql("ai_readonly"):
    print(sql)
```

### 5. Run the System

```bash
# Run demo
python main_pipeline.py

# Run Streamlit UI
streamlit run app.py
```

## Available Groq Models

| Model | Tier | Best For |
|-------|------|----------|
| `llama-3.1-8b-instant` | Fast | Routing, validation (low latency) |
| `llama-3.1-70b-versatile` | Balanced | SQL generation, storytelling |
| `mixtral-8x7b-32768` | Powerful | Complex reasoning tasks |

## Project Structure

```
Talk-to-Data/
├── layers/                      # 6 layer implementations
│   ├── layer1_semantic_cache.py
│   ├── layer2_intent_router.py
│   ├── layer3_tag.py
│   ├── layer4_multi_agent_sql.py
│   ├── layer5_secure_execution.py
│   ├── layer6_storyteller.py
│   └── groq_client.py           # Groq API wrapper
├── config/
│   └── config.yaml              # Configuration file
├── tests/
│   └── test_pipeline.py         # Unit tests
├── main_pipeline.py             # Main pipeline integration
├── app.py                       # Streamlit UI
├── requirements.txt
└── README.md
```

## Layer Details

### Layer 1: Semantic Cache
- Uses sentence-transformers for embedding
- Redis for storage with TTL
- Cosine similarity threshold (0.92)

### Layer 2: Intent Router
- Classifies queries into: `sql`, `rag`, `both`
- Uses fast Groq model (llama-3.1-8b-instant)
- LangGraph for orchestration

### Layer 3: Table Augmented Generation (TAG)
- ChromaDB for vector storage
- Dynamic schema retrieval
- Document RAG support

### Layer 4: Multi-Agent SQL Engine
- **Planner Agent**: Domain expert, creates execution plan (llama-3.1-70b)
- **Coder Agent**: Generates SQL query (llama-3.1-70b)
- **Validator Agent**: Security and syntax validation (llama-3.1-8b)
- sqlglot for syntax checking

### Layer 5: Secure Execution
- Read-only database role
- Parameterized queries
- Statement timeout protection

### Layer 6: Storyteller
- Natural language answer generation (llama-3.1-70b)
- Full lineage trace logging
- JSON audit trail

## Configuration

Edit `config/config.yaml` or set environment variables:

```yaml
semantic_cache:
  similarity_threshold: 0.92
  ttl_seconds: 3600

intent_router:
  model: llama-3.1-8b-instant

tag:
  top_k_schemas: 3

multi_agent_sql:
  planner_model: llama-3.1-70b-versatile
  coder_model: llama-3.1-70b-versatile
  validator_model: llama-3.1-8b-instant

storyteller:
  model: llama-3.1-70b-versatile
  temperature: 0.3
```

## API Usage

```python
from main_pipeline import AIQuerySystem

# Initialize
system = AIQuerySystem()

# Run query
response = system.run_pipeline("How many customers do we have?")

print(response.answer)           # Natural language answer
print(response.lineage.to_json()) # Full audit trail
```

## Running Tests

```bash
pytest tests/ -v
```
