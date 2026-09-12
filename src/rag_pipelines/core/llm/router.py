from functools import lru_cache
from typing import Any, TypeVar

import instructor
import litellm
from litellm import Router
from pydantic import BaseModel

from rag_pipelines.core.config import get_config
from rag_pipelines.core.logging import logger

ResponseT = TypeVar("ResponseT", bound=BaseModel)
ModelRegistry = dict[str, dict[str, Any]]

litellm.drop_params = True


def build_model_registry() -> tuple[ModelRegistry, str, list[str]]:
    """Build the default OpenRouter model registry."""
    config = get_config()

    if not config.OPENROUTER_API_KEY or not config.OPENROUTER_LLM_MODEL:
        raise RuntimeError(
            "OpenRouter is not configured. Set OPENROUTER_API_KEY and "
            "OPENROUTER_LLM_MODEL."
        )

    model = config.OPENROUTER_LLM_MODEL
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"

    registry: ModelRegistry = {
        "primary": {
            "model": model,
            "api_key": config.OPENROUTER_API_KEY,
            "api_base": config.OPENROUTER_API_URL.rstrip("/"),
        }
    }
    return registry, "primary", []


class ModelRouter:
    """LiteLLM model registry with fallback and Instructor validation."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        default_model: str | None = None,
        fallback_models: list[str] | None = None,
    ) -> None:
        if registry is None:
            registry, configured_default, configured_fallbacks = (
                build_model_registry()
            )
            default_model = default_model or configured_default
            if fallback_models is None:
                fallback_models = configured_fallbacks

        if not registry:
            raise ValueError("Model registry cannot be empty")

        self.registry = registry
        self.default_model = default_model or next(iter(registry))
        self.fallback_models = fallback_models or []

        if self.default_model not in registry:
            raise ValueError(
                f"Default model '{self.default_model}' is not registered"
            )

        unknown_fallbacks = set(self.fallback_models) - registry.keys()
        if unknown_fallbacks:
            raise ValueError(
                f"Fallback models are not registered: {sorted(unknown_fallbacks)}"
            )

        model_list = [
            {"model_name": name, "litellm_params": params}
            for name, params in registry.items()
        ]
        fallbacks = (
            [{self.default_model: self.fallback_models}]
            if self.fallback_models
            else []
        )

        self.router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=2,
        )
        self._instructor = instructor.from_litellm(self.router.completion)
        self._async_instructor = instructor.from_litellm(
            self.router.acompletion,
            async_client=True,
        )

        logger.info(
            "[ModelRouter] default=%s fallbacks=%s registered=%s",
            self.default_model,
            self.fallback_models,
            list(self.registry),
        )

    def completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ):
        """Return a standard LiteLLM chat completion."""
        return self.router.completion(
            model=self._resolve_model(model),
            messages=messages,
            **self._with_defaults(kwargs),
        )

    async def acompletion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ):
        """Return a standard asynchronous LiteLLM chat completion."""
        return await self.router.acompletion(
            model=self._resolve_model(model),
            messages=messages,
            **self._with_defaults(kwargs),
        )

    def structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[ResponseT],
        model: str | None = None,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> ResponseT:
        """Return output validated against a Pydantic model."""
        return self._instructor.chat.completions.create(
            model=self._resolve_model(model),
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **self._with_defaults(kwargs),
        )

    async def astructured(
        self,
        messages: list[dict[str, str]],
        response_model: type[ResponseT],
        model: str | None = None,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> ResponseT:
        """Return asynchronously validated Pydantic output."""
        return await self._async_instructor.chat.completions.create(
            model=self._resolve_model(model),
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **self._with_defaults(kwargs),
        )

    def _resolve_model(self, model: str | None) -> str:
        resolved = model or self.default_model
        if resolved not in self.registry:
            raise ValueError(f"Model '{resolved}' is not registered")
        return resolved

    def _with_defaults(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        config = get_config()
        if kwargs.get("temperature") is None:
            kwargs["temperature"] = config.MODEL_TEMPERATURE
        if kwargs.get("max_tokens") is None:
            kwargs["max_tokens"] = config.MODEL_MAX_TOKENS
        return kwargs


@lru_cache
def get_model_router() -> ModelRouter:
    return ModelRouter()
