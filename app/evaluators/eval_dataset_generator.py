# app/evaluators/eval_dataset_generator.py
"""
app/evaluators/eval_dataset_generator.py

Automatic evaluation dataset generator.

Responsibilities:
  - Accept a list of Chunk objects produced by the ingestion pipeline.
  - Call the LLM once per chunk to generate a (question, reference) pair.
  - Run each generated question through the RAG pipeline to obtain the
    system's actual answer and the retrieved contexts.
  - Write complete JSONL samples to data/evaluation_dataset/<stem>.jsonl.

Fix #22 — Auto-generated eval datasets are incomplete for RAGAS:
  Previously, `answer` and `contexts` were written as empty string/list and
  described as "populated at eval time". Nothing ever populated them. Running
  /evaluate on those files produced null metrics for faithfulness and
  answer_relevancy (which require a real answer) and misleading scores for
  context_precision / context_recall (which require real retrieved contexts).

  Fixed behaviour:
  - After generating a question via the LLM, the generator immediately runs
    that question through ChromaManager.similarity_search() to retrieve the
    top-K context chunks, then through RAGService.generate_answer() to obtain
    the system's answer.
  - Both `answer` (str) and `contexts` (list[str]) are written fully populated
    into every JSONL sample.
  - If the RAG step fails for a single chunk, that sample is skipped (same
    non-fatal contract as before). The question/reference generation and RAG
    answer steps are separated so a RAG failure does not waste the LLM call
    that produced the question.

JSONL schema (one JSON object per line):
    {
        "question":      str,        # LLM-generated question about the chunk
        "answer":        str,        # RAG-pipeline answer (populated at generation)
        "contexts":      list[str],  # retrieved chunk texts (populated at generation)
        "reference":     str,        # LLM-generated ground-truth answer
        "chunk_id":      str,        # source chunk UUID (traceability)
        "chunk_index":   int,        # position within source document
        "document_name": str         # source document filename
    }

Design decisions:
  1. ChromaManager and RAGService are injected via constructor, not
     instantiated internally. This keeps the class testable and avoids
     circular import issues (RAGService imports ChromaManager).
  2. Generation still runs per-chunk sequentially. The non-fatal per-chunk
     contract is preserved — one failure never aborts the batch.
  3. Chunks shorter than MIN_CHUNK_LENGTH are still skipped.
  4. File I/O still uses append mode. Re-ingesting a document adds new
     samples; callers wanting a clean slate should delete the JSONL first.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.models.documents import Chunk
from app.utils.config import get_settings
from app.utils.llm_factory import create_llm_with_fallback
from app.utils.logger import get_logger

if TYPE_CHECKING:
    # Avoid circular import at runtime; only used for type hints.
    from app.services.rag_service import RAGService
    from app.vectorstore.chroma_manager import ChromaManager

logger = get_logger(__name__)

# Chunks shorter than this threshold are skipped during generation.
MIN_CHUNK_LENGTH: int = 100

# Output directory for generated evaluation datasets.
_EVAL_DATASET_DIR = Path("data/evaluation_dataset")

# Number of top chunks to retrieve when populating `contexts`.
_RETRIEVAL_TOP_K: int = 3

_SYSTEM_PROMPT = """\
You are an expert at creating evaluation datasets for Retrieval-Augmented Generation (RAG) systems.

Given a passage of text, generate exactly ONE question that:
- Can be answered using ONLY the information in the passage.
- Is specific and unambiguous.
- Would be a realistic question a user might ask about a document.

Then provide the correct reference answer based solely on the passage.

Respond with ONLY a JSON object in this exact format, with no preamble, no markdown, and no trailing text:
{"question": "<your question here>", "reference": "<your reference answer here>"}
"""


class EvalDatasetGenerator:
    """
    Generates complete evaluation Q&A samples from ingested document chunks.

    Each call to generate_from_chunks() produces JSONL samples with all four
    RAGAS-required fields populated: question, answer, contexts, reference.

    Args:
        chroma_manager: Shared ChromaManager instance for similarity search.
        rag_service:    Shared RAGService instance for answer generation.

    Usage (called from the upload pipeline after ChromaDB storage):
        generator = EvalDatasetGenerator(
            chroma_manager=chroma_manager,
            rag_service=rag_service,
        )
        written = generator.generate_from_chunks(chunks, "contract_a.pdf")
        # → 12  (number of fully populated samples written)
    """

    def __init__(
        self,
        chroma_manager: "ChromaManager",
        rag_service: "RAGService",
    ) -> None:
        settings = get_settings()

        self._chroma_manager = chroma_manager
        self._rag_service = rag_service

        self._llm = create_llm_with_fallback(
            temperature=0.2,
            max_tokens=512,
        )

        self._output_dir = _EVAL_DATASET_DIR
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EvalDatasetGenerator initialised | model=%s | output_dir=%s",
            settings.LLM_MODEL,
            self._output_dir,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_from_chunks(
        self,
        chunks: list[Chunk],
        document_name: str,
    ) -> int:
        """
        Generate fully populated evaluation Q&A samples and write them to disk.

        For each eligible chunk:
          1. Ask the LLM to generate a (question, reference) pair.
          2. Run the question through ChromaDB to retrieve real contexts.
          3. Run the question through RAGService to get the system's answer.
          4. Write the complete sample to JSONL.

        Args:
            chunks:        Chunk objects produced by DocumentChunker.
            document_name: Original filename (e.g. "contract_a.pdf").

        Returns:
            Number of fully populated samples written to disk.
        """
        output_path = self._resolve_output_path(document_name)
        eligible = [c for c in chunks if len(c.chunk_text) >= MIN_CHUNK_LENGTH]
        skipped = len(chunks) - len(eligible)

        if skipped:
            logger.debug(
                "Skipped %d short chunk(s) from '%s' (below %d chars).",
                skipped,
                document_name,
                MIN_CHUNK_LENGTH,
            )

        if not eligible:
            logger.warning(
                "No eligible chunks found for '%s' — eval dataset not generated.",
                document_name,
            )
            return 0

        logger.info(
            "Generating eval samples | document='%s' | eligible_chunks=%d | output='%s'",
            document_name,
            len(eligible),
            output_path,
        )

        written = 0
        for chunk in eligible:
            sample = self._build_sample(chunk)
            if sample is None:
                continue
            try:
                self._append_sample(output_path, sample)
                written += 1
            except OSError as exc:
                logger.error(
                    "Failed to write eval sample for chunk '%s': %s",
                    chunk.chunk_id,
                    exc,
                )

        logger.info(
            "Eval dataset generation complete | document='%s' | written=%d/%d",
            document_name,
            written,
            len(eligible),
        )
        return written

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_sample(self, chunk: Chunk) -> dict[str, Any] | None:
        """
        Build one complete JSONL sample for a single chunk.

        Steps:
          1. Generate question + reference via LLM.
          2. Retrieve top-K context chunks from ChromaDB for the question.
          3. Generate system answer via RAGService using those contexts.

        Returns:
            Complete sample dict, or None if any step fails.
        """
        # Step 1: generate question and reference answer
        qa_pair = self._generate_qa_pair(chunk)
        if qa_pair is None:
            return None

        question = qa_pair["question"]
        reference = qa_pair["reference"]

        # Step 2: retrieve contexts for this question
        # Fix #22: contexts must be real retrieved text, not an empty list.
        try:
            retrieval_results = self._chroma_manager.similarity_search(
                query=question,
                top_k=_RETRIEVAL_TOP_K,
            )
            contexts: list[str] = [r["chunk_text"] for r in retrieval_results]
        except Exception as exc:
            logger.error(
                "Retrieval failed for question from chunk '%s': %s",
                chunk.chunk_id,
                exc,
            )
            return None

        if not contexts:
            logger.warning(
                "No contexts retrieved for question from chunk '%s' — skipping sample.",
                chunk.chunk_id,
            )
            return None

        # Step 3: generate the system's answer via RAGService
        # Fix #22: answer must be the actual RAG output, not an empty string.
        try:
            rag_result = self._rag_service.query(
                question=question,
                top_k=_RETRIEVAL_TOP_K,
            )
            answer: str = rag_result.answer
        except Exception as exc:
            logger.error(
                "RAG answer generation failed for chunk '%s': %s",
                chunk.chunk_id,
                exc,
            )
            return None

        if not answer.strip():
            logger.warning(
                "RAGService returned empty answer for chunk '%s' — skipping sample.",
                chunk.chunk_id,
            )
            return None

        return {
            "question": question,
            "answer": answer,               # real RAG answer — not ""
            "contexts": contexts,           # real retrieved texts — not []
            "reference": reference,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "document_name": chunk.document_name,
        }

    def _generate_qa_pair(self, chunk: Chunk) -> dict[str, str] | None:
        """
        Call the LLM to generate a (question, reference) pair for one chunk.

        Args:
            chunk: A single Chunk with chunk_text content.

        Returns:
            Dict with "question" and "reference" keys, or None on failure.
        """
        user_message = (
            "Generate a question and reference answer for the following passage:\n\n"
            f"{chunk.chunk_text}"
        )

        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=user_message),
                ]
            )
        except Exception as exc:
            logger.error(
                "LLM call failed for chunk '%s' (document='%s'): %s",
                chunk.chunk_id,
                chunk.document_name,
                exc,
            )
            return None

        raw_text: str = response.content if isinstance(response.content, str) else ""
        return self._parse_llm_response(raw_text, chunk.chunk_id)

    def _parse_llm_response(
        self,
        raw_text: str,
        chunk_id: str,
    ) -> dict[str, str] | None:
        """
        Parse the LLM's JSON response into a question/reference dict.

        Handles two cases:
          1. The model returned clean JSON as instructed.
          2. The model wrapped JSON in markdown code fences.

        Returns:
            Dict with "question" and "reference" keys, or None if parsing failed.
        """
        text = raw_text.strip()

        # Strip markdown code fences if the model disobeyed the system prompt.
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "LLM returned non-JSON for chunk '%s'. Response: %r. Error: %s",
                chunk_id,
                raw_text[:200],
                exc,
            )
            return None

        if "question" not in data or "reference" not in data:
            logger.warning(
                "LLM response missing required fields for chunk '%s'. Got keys: %s",
                chunk_id,
                list(data.keys()),
            )
            return None

        question = str(data["question"]).strip()
        reference = str(data["reference"]).strip()

        if not question or not reference:
            logger.warning(
                "LLM returned empty question or reference for chunk '%s'.",
                chunk_id,
            )
            return None

        return {"question": question, "reference": reference}

    def _resolve_output_path(self, document_name: str) -> Path:
        """
        Derive the JSONL output path from the source document name.

        "contract_a.pdf"   → data/evaluation_dataset/contract_a.jsonl
        "report 2024.docx" → data/evaluation_dataset/report 2024.jsonl
        """
        stem = Path(document_name).stem
        return self._output_dir / f"{stem}.jsonl"

    def _append_sample(self, output_path: Path, sample: dict[str, Any]) -> None:
        """
        Append a single sample as a JSON line to the output file.

        Uses append mode so existing samples are never overwritten.
        """
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")