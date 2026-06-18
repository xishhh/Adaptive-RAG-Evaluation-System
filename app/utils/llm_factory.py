"""
app/utils/llm_factory.py

Creates a LangChain Runnable that calls multiple ChatOpenAI models with
exponential-backoff retry and automatic fallback.

Flow per invoke/stream call:
  1. Try the primary model.
  2. On rate-limit / API error → wait (2s, 4s, 8s) and retry up to 3 times.
  3. If all retries fail → move to the next model in the list.
  4. Repeat until a model succeeds.
  5. If ALL models fail → raise the last exception.

Usage:
    llm = create_llm_with_fallback(temperature=0.0, max_tokens=1024)
    response = llm.invoke(messages)
    for chunk in llm.stream(messages): ...
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import openai
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.utils.config import get_settings

logger = logging.getLogger(__name__)

# Exceptions that trigger a retry (with backoff) then fallback.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APIError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds; doubled each retry → 2, 4, 8


def _parse_models(models_str: str) -> list[str]:
    """Parse a comma-separated model string into a non-empty list."""
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    if models:
        return models
    return [get_settings().LLM_MODEL]


def _build_llm(model: str, **common: Any) -> ChatOpenAI:
    """Build a single ChatOpenAI instance with the given overrides."""
    return ChatOpenAI(model=model, **common)


def _should_retry(exc: Exception) -> bool:
    """Return True if *exc* is a retryable API error."""
    return isinstance(exc, _RETRYABLE)


class _RetryFallbackLLM(Runnable):
    """
    Runnable that wraps multiple ChatOpenAI instances with retry + fallback.

    For each model (in order):
      - Attempt the call.
      - On a retryable error → sleep with exponential backoff, retry.
      - After ``_MAX_RETRIES`` failed attempts → log a warning and try the
        next model.

    If every model in the list fails, the last exception is re-raised.
    """

    def __init__(self, models: list[ChatOpenAI]) -> None:
        self._models = models

    # ------------------------------------------------------------------
    # invoke
    # ------------------------------------------------------------------
    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        last_exc: Exception | None = None

        for idx, llm in enumerate(self._models):
            model_name = llm.model  # type: ignore[attr-defined]
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    return llm.invoke(input, config=config, **kwargs)
                except _RETRYABLE as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        delay = _BASE_DELAY * (2 ** (attempt - 1))
                        logger.warning(
                            "Model '%s' attempt %d/%d failed (%s). "
                            "Retrying in %.0fs …",
                            model_name,
                            attempt,
                            _MAX_RETRIES,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "Model '%s' exhausted %d retries. "
                            "Falling back to next model …",
                            model_name,
                            _MAX_RETRIES,
                        )
                except Exception as exc:
                    # Non-retryable error → skip this model immediately.
                    last_exc = exc
                    logger.warning(
                        "Model '%s' failed with non-retryable error: %s. "
                        "Falling back …",
                        model_name,
                        exc,
                    )
                    break  # to next model

        # All models exhausted.
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # stream
    # ------------------------------------------------------------------
    def stream(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        last_exc: Exception | None = None

        for idx, llm in enumerate(self._models):
            model_name = llm.model  # type: ignore[attr-defined]
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    yield from llm.stream(input, config=config, **kwargs)
                    return  # stream completed successfully
                except _RETRYABLE as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        delay = _BASE_DELAY * (2 ** (attempt - 1))
                        logger.warning(
                            "Model '%s' stream attempt %d/%d failed (%s). "
                            "Retrying in %.0fs …",
                            model_name,
                            attempt,
                            _MAX_RETRIES,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning(
                            "Model '%s' stream exhausted %d retries. "
                            "Falling back …",
                            model_name,
                            _MAX_RETRIES,
                        )
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Model '%s' stream failed: %s. Falling back …",
                        model_name,
                        exc,
                    )
                    break  # to next model

        assert last_exc is not None
        raise last_exc


def create_llm_with_fallback(**kwargs: Any) -> Runnable:
    """
    Build a Runnable with retry + fallback across configured models.

    Keyword arguments (``temperature``, ``max_tokens``, …) are forwarded to
    every ``ChatOpenAI`` instance in the chain.

    Returns
    -------
    Runnable
        A ``_RetryFallbackLLM`` instance when multiple models are configured,
        or a plain ``ChatOpenAI`` if only one model is given.  Call
        ``.invoke(messages)`` / ``.stream(messages)`` as usual.
    """
    settings = get_settings()
    models = _parse_models(settings.LLM_MODELS)

    common: dict[str, Any] = {
        "openai_api_key": settings.OPENAI_API_KEY,
        "openai_api_base": settings.OPENAI_API_BASE,
        **kwargs,
    }

    instances = [_build_llm(m, **common) for m in models]

    logger.info(
        "LLM chain: %s  (retries=%d, base_delay=%.0fs)",
        " → ".join(models),
        _MAX_RETRIES,
        _BASE_DELAY,
    )

    if len(instances) == 1:
        return instances[0]

    return _RetryFallbackLLM(instances)
