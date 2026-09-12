from typing import TypeVar

from pydantic import BaseModel

from rag_pipelines.core.llm.router import get_model_router

T = TypeVar("T", bound=BaseModel)


def invoke_llm(prompt: str, max_tokens: int | None = None) -> str:
    """Invoke the default routed model and return its text."""
    response = get_model_router().completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def invoke_structured(
    prompt: str,
    response_model: type[T],
    max_tokens: int | None = None,
) -> T:
    """Invoke the default routed model with Pydantic validation."""
    return get_model_router().structured(
        messages=[{"role": "user", "content": prompt}],
        response_model=response_model,
        max_tokens=max_tokens,
    )
