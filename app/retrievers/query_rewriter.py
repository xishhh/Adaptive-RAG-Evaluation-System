"""
app/retrievers/query_rewriter.py

Rewrites a low-confidence query into an expanded set of search terms
to improve vector-store recall on re-retrieval.

Architecture reference: Component 6 (Query Rewriter) in SYSTEM_ARCHITECTURE.md.

Activation condition:
    Called by AdaptiveRAGService only when ConfidenceEvaluator returns
    LOW_CONFIDENCE. Never called on GOOD_CONTEXT paths.

Rewriting strategy:
    The LLM is prompted to generate synonyms, alternative phrasings, and
    related domain terms — all on a single line as a comma-separated list.
    This expanded string is passed directly to Retriever.retrieve() as
    the new query, broadening the embedding search space.

Example:
    Original:  "What penalties exist?"
    Rewritten: "Penalty clauses, liquidated damages, delayed delivery
                penalties, financial penalties, breach of contract
                consequences"

Design decisions:
  1. The rewritten query is a single string, not a list of sub-queries.
     ChromaDB performs one similarity search per call, so a single
     dense string is the right unit.
  2. On LLM failure, the original query is returned unchanged. This is
     the safe fallback: re-retrieval with the original query is no worse
     than the first retrieval and avoids a hard failure.
  3. Constructor injection for the LLM keeps this testable in isolation.
"""

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
    """
    Rewrites a user query into an expanded, retrieval-optimised search string.

    Args:
        llm: An initialised LangChain language model instance.
             Can be a ``ChatOpenAI`` or a runnable with fallbacks.

    Usage:
        rewriter = QueryRewriter(llm=create_llm_with_fallback())
        expanded = rewriter.rewrite("What penalties exist?")
        # → "Penalty clauses, liquidated damages, delayed delivery penalties …"
    """

    def __init__(self, llm: BaseLanguageModel) -> None:
        self._llm = llm

    def rewrite(self, query: str) -> str:
        """
        Expand the query into richer search terms.

        Args:
            query: The original user query that yielded LOW_CONFIDENCE retrieval.

        Returns:
            An expanded query string for re-retrieval.
            Falls back to the original query on LLM failure.
        """
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

            logger.info(
                "Query rewritten | original='%s' | rewritten='%s'",
                query[:80],
                rewritten[:120],
            )
            return rewritten

        except Exception as exc:
            # Safe fallback: re-retrieval with the original query.
            logger.warning(
                "QueryRewriter failed — falling back to original query. Error: %s",
                exc,
            )
            return query