"""
app/retrievers/query_classifier.py

Classifies an incoming user query to decide whether vector-store
retrieval is needed or whether the LLM can answer directly.

Architecture reference: Component 3 (Query Processing) in SYSTEM_ARCHITECTURE.md.

Classification outcomes:
    DIRECT_LLM      — General knowledge question; no retrieval needed.
                      Example: "What is FastAPI?"
    KNOWLEDGE_QUERY — Requires document context from the vector store.
                      Example: "What is the termination clause in Contract A?"

Design decisions:
  1. LLM-based classification using a structured prompt that forces a
     single JSON key. This is more reliable than keyword heuristics for
     domain-agnostic queries.
  2. The prompt is strict: the model must respond with ONLY a JSON object
     {"query_type": "DIRECT_LLM"} or {"query_type": "KNOWLEDGE_QUERY"}.
     Any parse failure defaults to KNOWLEDGE_QUERY (safe fallback —
     it's always better to retrieve unnecessarily than to skip retrieval).
  3. Constructor injection for the LLM makes the classifier testable
     without hitting the API.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

QueryType = Literal["DIRECT_LLM", "KNOWLEDGE_QUERY"]

_SYSTEM_PROMPT = """You are a query routing assistant for a document question-answering system.

Your ONLY job is to decide whether a user query requires searching a document knowledge base.

Rules:
- KNOWLEDGE_QUERY: The question asks about specific documents, contracts, reports, policies,
  people, places, events, or any information that would realistically be stored in a
  private document corpus. When in doubt, choose KNOWLEDGE_QUERY.
- DIRECT_LLM: The question is a general knowledge question that any well-trained language
  model can answer without documents. Examples: definitions, concepts, how-to explanations,
  math problems, general facts.

Respond with ONLY a valid JSON object and nothing else. No explanation. No markdown.
Format: {"query_type": "KNOWLEDGE_QUERY"} or {"query_type": "DIRECT_LLM"}
"""


class QueryClassifier:
    """
    Classifies a user query into DIRECT_LLM or KNOWLEDGE_QUERY.

    Args:
        llm: An initialised LangChain language model instance.
             Can be a ``ChatOpenAI`` or a runnable with fallbacks.

    Usage:
        classifier = QueryClassifier(llm=create_llm_with_fallback())
        query_type = classifier.classify("What is the termination clause?")
        # → "KNOWLEDGE_QUERY"
    """

    def __init__(self, llm: BaseLanguageModel) -> None:
        self._llm = llm

    def classify(self, query: str) -> QueryType:
        """
        Classify the query and return the routing decision.

        Args:
            query: The user's natural-language question.

        Returns:
            "KNOWLEDGE_QUERY" or "DIRECT_LLM".
            Falls back to "KNOWLEDGE_QUERY" on any parse error.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        logger.info("Classifying query: '%s'", query[:80])

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}"),
        ]

        try:
            response = self._llm.invoke(messages)
            raw = response.content.strip()
            parsed = json.loads(raw)
            query_type: QueryType = parsed["query_type"]

            if query_type not in ("DIRECT_LLM", "KNOWLEDGE_QUERY"):
                raise ValueError(f"Unexpected query_type value: '{query_type}'")

            logger.info("Query classified as: %s", query_type)
            return query_type

        except Exception as exc:
            # Safe fallback: always retrieve rather than skip retrieval.
            logger.warning(
                "QueryClassifier failed to parse LLM response — "
                "defaulting to KNOWLEDGE_QUERY. Error: %s",
                exc,
            )
            return "KNOWLEDGE_QUERY"

