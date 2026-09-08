# rag-pipelines

RAG pipeline library for document ingestion and vector store operations.

## Install

```bash
# Core library only
pip install rag-pipelines

# With AWS (Bedrock, S3, S3 Vectors)
pip install "rag-pipelines[aws]"

# With document parsers (PDF, Excel, Word, etc.)
pip install "rag-pipelines[parsers]"

# With FastAPI server
pip install "rag-pipelines[server]"

# Everything
pip install "rag-pipelines[all]"
```

### Development (editable install)

```bash
git clone https://github.com/louisaxelta/rag-pipelines.git
cd rag-pipelines
pip install -e ".[all]"
cp .env.example .env
```

## Run the API server

```bash
rag-pipelines
```

Or:

```bash
uvicorn rag_pipelines.app.main:app --reload
```

Health check: `GET http://localhost:8000/health`

## Project structure

```text
src/rag_pipelines/
├── app/            # FastAPI application
├── core/           # Ingestion, workflow, retrieval
└── integrations/   # Optional third-party tools (Langfuse)
```

## Configuration

Copy `.env.example` to `.env` and fill in your values. See `rag_pipelines.core.config.Config` for all options.