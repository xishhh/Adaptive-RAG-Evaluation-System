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

_RETRYABLE = (
    openai.RateLimitError,
    openai.APIError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_MAX_RETRIES = 10
_BASE_DELAY = 5.0
_MAX_BACKOFF = 120.0


def _parse_models(models_str: str) -> list[str]:
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    if models:
        return models
    return [get_settings().LLM_MODEL]


def _build_llm(model: str, **common: Any) -> ChatOpenAI:
    return ChatOpenAI(model=model, **common)


def _extract_retry_after(exc: Exception) -> float | None:
    try:
        body = getattr(exc, "body", None)
        if not isinstance(body, dict):
            return None
        return body.get("error", {}).get("metadata", {}).get("retry_after_seconds")
    except Exception:
        return None


def _compute_delay(attempt: int, exc: Exception) -> float:
    retry_after = _extract_retry_after(exc)
    if retry_after is not None:
        return min(retry_after, _MAX_BACKOFF)
    return min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_BACKOFF)


class _RetryFallbackLLM(Runnable):
    def __init__(self, models: list[ChatOpenAI]) -> None:
        self._models = models

    def invoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        last_exc: Exception | None = None

        for idx, llm in enumerate(self._models):
            model_name = llm.model_name
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    return llm.invoke(input, config=config, **kwargs)
                except _RETRYABLE as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        delay = _compute_delay(attempt, exc)
                        logger.warning("Model '%s' attempt %d/%d failed (%s). Retrying in %.0fs ...", model_name, attempt, _MAX_RETRIES, exc, delay)
                        time.sleep(delay)
                    else:
                        logger.warning("Model '%s' exhausted %d retries. Falling back to next model ...", model_name, _MAX_RETRIES)
                except Exception as exc:
                    last_exc = exc
                    logger.warning("Model '%s' failed with non-retryable error: %s. Falling back ...", model_name, exc)
                    break

        assert last_exc is not None
        raise last_exc

    def stream(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[BaseMessage]:
        last_exc: Exception | None = None

        for idx, llm in enumerate(self._models):
            model_name = llm.model_name
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    yield from llm.stream(input, config=config, **kwargs)
                    return
                except _RETRYABLE as exc:
                    last_exc = exc
                    if attempt < _MAX_RETRIES:
                        delay = _compute_delay(attempt, exc)
                        logger.warning("Model '%s' stream attempt %d/%d failed (%s). Retrying in %.0fs ...", model_name, attempt, _MAX_RETRIES, exc, delay)
                        time.sleep(delay)
                    else:
                        logger.warning("Model '%s' stream exhausted %d retries. Falling back ...", model_name, _MAX_RETRIES)
                except Exception as exc:
                    last_exc = exc
                    logger.warning("Model '%s' stream failed: %s. Falling back ...", model_name, exc)
                    break

        assert last_exc is not None
        raise last_exc


def create_llm_with_fallback(**kwargs: Any) -> Runnable:
    settings = get_settings()
    models = _parse_models(settings.LLM_MODELS)

    common: dict[str, Any] = {
        "openai_api_key": settings.OPENAI_API_KEY,
        "openai_api_base": settings.OPENAI_API_BASE,
        **kwargs,
    }

    instances = [_build_llm(m, **common) for m in models]

    logger.info("LLM chain: %s  (retries=%d, base_delay=%.0fs)", " → ".join(models), _MAX_RETRIES, _BASE_DELAY)

    if len(instances) == 1:
        return instances[0]

    return _RetryFallbackLLM(instances)
