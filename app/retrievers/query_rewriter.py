from __future__ import annotations

import logging

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a search query expansion specialist for a document retrieval system.

Your job is to rewrite a user's query into an expanded set of search terms that will
retrieve MORE relevant document chunks from a vector database.

Rules:
- Include the original query's core intent.
- Add synonyms, related legal/business/technical terms, and alternative phrasings.
- Expand acronyms if applicable.
- Output a SINGLE line of comma-separated search terms. No bullet points, no numbering,
  no explanation, no markdown. Just the expanded terms on one line.

Example input:  "What penalties exist for delayed delivery?"
Example output: "Penalty clauses, liquidated damages, delayed delivery penalties, delay charges, late delivery fees, financial consequences, breach of contract, delivery schedule violations"
"""


class QueryRewriter:
    def __init__(self, llm: BaseLanguageModel) -> None:
        self._llm = llm

    def rewrite(self, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        logger.info("Rewriting query: '%s'", query[:80])

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Query to expand: {query}"),
        ]

        try:
            response = self._llm.invoke(messages)
            rewritten = response.content.strip()
            if not rewritten:
                raise ValueError("LLM returned an empty rewrite.")
            logger.info("Query rewritten | original='%s' | rewritten='%s'", query[:80], rewritten[:120])
            return rewritten
        except Exception as exc:
            logger.warning("QueryRewriter failed — falling back to original query. Error: %s", exc)
            return query
