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
    from app.services.rag_service import RAGService
    from app.vectorstore.chroma_manager import ChromaManager

logger = get_logger(__name__)

MIN_CHUNK_LENGTH: int = 100
_EVAL_DATASET_DIR = Path("data/evaluation_dataset")
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

    def generate_from_chunks(
        self,
        chunks: list[Chunk],
        document_name: str,
    ) -> int:
        output_path = self._resolve_output_path(document_name)
        eligible = [c for c in chunks if len(c.chunk_text) >= MIN_CHUNK_LENGTH]
        skipped = len(chunks) - len(eligible)

        if skipped:
            logger.debug("Skipped %d short chunk(s) from '%s' (below %d chars).", skipped, document_name, MIN_CHUNK_LENGTH)

        if not eligible:
            logger.warning("No eligible chunks found for '%s' — eval dataset not generated.", document_name)
            return 0

        logger.info("Generating eval samples | document='%s' | eligible_chunks=%d | output='%s'", document_name, len(eligible), output_path)

        written = 0
        for chunk in eligible:
            sample = self._build_sample(chunk)
            if sample is None:
                continue
            try:
                self._append_sample(output_path, sample)
                written += 1
            except OSError as exc:
                logger.error("Failed to write eval sample for chunk '%s': %s", chunk.chunk_id, exc)

        logger.info("Eval dataset generation complete | document='%s' | written=%d/%d", document_name, written, len(eligible))
        return written

    def _build_sample(self, chunk: Chunk) -> dict[str, Any] | None:
        qa_pair = self._generate_qa_pair(chunk)
        if qa_pair is None:
            return None

        question = qa_pair["question"]
        reference = qa_pair["reference"]

        try:
            retrieval_results = self._chroma_manager.similarity_search(
                query=question,
                top_k=_RETRIEVAL_TOP_K,
            )
            contexts: list[str] = [r["chunk_text"] for r in retrieval_results]
        except Exception as exc:
            logger.error("Retrieval failed for question from chunk '%s': %s", chunk.chunk_id, exc)
            return None

        if not contexts:
            logger.warning("No contexts retrieved for question from chunk '%s' — skipping.", chunk.chunk_id)
            return None

        try:
            rag_result = self._rag_service.query(
                question=question,
                top_k=_RETRIEVAL_TOP_K,
            )
            answer: str = rag_result.answer
        except Exception as exc:
            logger.error("RAG answer generation failed for chunk '%s': %s", chunk.chunk_id, exc)
            return None

        if not answer.strip():
            logger.warning("RAGService returned empty answer for chunk '%s' — skipping.", chunk.chunk_id)
            return None

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": reference,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "document_name": chunk.document_name,
        }

    def _generate_qa_pair(self, chunk: Chunk) -> dict[str, str] | None:
        user_message = f"Generate a question and reference answer for the following passage:\n\n{chunk.chunk_text}"

        try:
            response = self._llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ])
        except Exception as exc:
            logger.error("LLM call failed for chunk '%s' (document='%s'): %s", chunk.chunk_id, chunk.document_name, exc)
            return None

        raw_text: str = response.content if isinstance(response.content, str) else ""
        return self._parse_llm_response(raw_text, chunk.chunk_id)

    def _parse_llm_response(self, raw_text: str, chunk_id: str) -> dict[str, str] | None:
        text = raw_text.strip()

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned non-JSON for chunk '%s'. Response: %r. Error: %s", chunk_id, raw_text[:200], exc)
            return None

        if "question" not in data or "reference" not in data:
            logger.warning("LLM response missing required fields for chunk '%s'. Got keys: %s", chunk_id, list(data.keys()))
            return None

        question = str(data["question"]).strip()
        reference = str(data["reference"]).strip()
        if not question or not reference:
            logger.warning("LLM returned empty question or reference for chunk '%s'.", chunk_id)
            return None

        return {"question": question, "reference": reference}

    def _resolve_output_path(self, document_name: str) -> Path:
        stem = Path(document_name).stem
        return self._output_dir / f"{stem}.jsonl"

    def _append_sample(self, output_path: Path, sample: dict[str, Any]) -> None:
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
