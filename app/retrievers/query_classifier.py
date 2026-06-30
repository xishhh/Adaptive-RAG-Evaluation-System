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
    def __init__(self, llm: BaseLanguageModel) -> None:
        self._llm = llm

    def classify(self, query: str) -> QueryType:
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
            logger.warning("QueryClassifier failed — defaulting to KNOWLEDGE_QUERY. Error: %s", exc)
            return "KNOWLEDGE_QUERY"
