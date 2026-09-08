from fastapi import FastAPI

from rag_pipelines import __version__
from rag_pipelines.app.api.routes import router as documents_router

app = FastAPI(
    title="RAG Pipelines",
    version=__version__,
    description="Document ingestion and vector store pipeline service",
)

app.include_router(documents_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "rag_pipelines.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
