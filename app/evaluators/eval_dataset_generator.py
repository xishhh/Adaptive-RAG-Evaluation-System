# app/evaluators/eval_dataset_generator.py
"""
app/evaluators/eval_dataset_generator.py

Automatic evaluation dataset generator.

Responsibilities:
  - Accept a list of Chunk objects produced by the ingestion pipeline.
  - Call the LLM once per chunk to generate a (question, reference_answer) pair.
  - Write each sample to a per-document JSONL file under
    data/evaluation_dataset/<document_stem>.jsonl.
  - Leave the `answer` and `contexts` fields empty — they are populated
    at evaluation time by RagasEvaluator, not at ingestion time.

JSONL schema (one JSON object per line):
    {
        "question":   str,        # LLM-generated question about the chunk
        "answer":     "",         # populated at eval time
        "contexts":   [],         # populated at eval time
        "reference":  str,        # LLM-generated ground-truth answer
        "chunk_id":   str,        # source chunk UUID (traceability)
        "chunk_index": int,       # position within source document
        "document_name": str      # source document filename
    }

Design decisions:
  1. One JSONL file per source document, named by the document stem
     (e.g. "contract_a.pdf" → "contract_a.jsonl"). This allows
     per-document or global evaluation without restructuring.
  2. Generation failures are caught per-chunk. One bad LLM call never
     aborts the batch; the sample is skipped and the error is logged.
  3. Chunks shorter than MIN_CHUNK_LENGTH are skipped — they are
     typically headers, footers, or table fragments that produce
     degenerate questions and pollute the evaluation set.
  4. The LLM is prompted with a strict schema-response instruction.
     Responses are parsed as JSON; malformed responses are logged and
     skipped rather than raising.
  5. File I/O uses append mode so re-ingesting a document adds new
     samples without destroying existing ones. Callers that want a
     clean slate should delete the JSONL file before re-ingestion.
  6. This class has no dependency on ChromaDB, the retriever, or any
     Phase 4+ component. It touches only the LLM and the filesystem.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.documents import Chunk
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Chunks shorter than this threshold are skipped during generation.
# Very short chunks (page numbers, headings, single table cells) yield
# low-quality questions that degrade evaluation signal.
MIN_CHUNK_LENGTH: int = 100

# Output directory for generated evaluation datasets.
# Matches the project structure: data/evaluation_dataset/
_EVAL_DATASET_DIR = Path("data/evaluation_dataset")

# System prompt sent once per LLM call to set response format.
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
    Generates evaluation Q&A samples from ingested document chunks.

    Each call to generate_from_chunks() appends new JSONL samples to a
    per-document file under data/evaluation_dataset/.

    Usage (called from the upload pipeline after ChromaDB storage):
        generator = EvalDatasetGenerator()
        written = generator.generate_from_chunks(chunks, "contract_a.pdf")
        # → 12  (number of samples successfully written)
    """

    def __init__(self) -> None:
        settings = get_settings()

        self._llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.2,  # slight creativity for question variety
            max_tokens=512,   # questions + answers are short; cap spend
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
        Generate evaluation Q&A samples for all eligible chunks and write
        them to data/evaluation_dataset/<document_stem>.jsonl.

        Args:
            chunks:        Chunk objects produced by DocumentChunker.
            document_name: Original filename (e.g. "contract_a.pdf").
                           Used to derive the output JSONL filename.

        Returns:
            Number of samples successfully written to disk.
            Returns 0 if all chunks were skipped or all LLM calls failed.

        Note:
            This method never raises. All errors are caught, logged, and
            counted. A zero return value indicates the caller should check
            the logs for generation warnings.
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
            sample = self._generate_sample(chunk)
            if sample is None:
                continue  # error already logged inside _generate_sample
            try:
                self._append_sample(output_path, sample)
                written += 1
            except OSError as exc:
                logger.error(
                    "Failed to write eval sample for chunk '%s' to '%s': %s",
                    chunk.chunk_id,
                    output_path,
                    exc,
                )

        logger.info(
            "Eval dataset generation complete | document='%s' | written=%d/%d samples",
            document_name,
            written,
            len(eligible),
        )
        return written

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _resolve_output_path(self, document_name: str) -> Path:
        """
        Derive the JSONL output path from the source document name.

        "contract_a.pdf"   → data/evaluation_dataset/contract_a.jsonl
        "report 2024.docx" → data/evaluation_dataset/report 2024.jsonl

        Args:
            document_name: Original document filename with extension.

        Returns:
            Absolute Path to the target JSONL file.
        """
        stem = Path(document_name).stem
        return self._output_dir / f"{stem}.jsonl"

    def _generate_sample(self, chunk: Chunk) -> dict[str, Any] | None:
        """
        Call the LLM to generate a question/reference pair for one chunk.

        Args:
            chunk: A single Chunk object with chunk_text content.

        Returns:
            A dict matching the JSONL schema, or None if generation failed.
        """
        user_message = (
            f"Generate a question and reference answer for the following passage:\n\n"
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
        parsed = self._parse_llm_response(raw_text, chunk.chunk_id)
        if parsed is None:
            return None

        return {
            "question": parsed["question"],
            "answer": "",          # populated at evaluation time
            "contexts": [],        # populated at evaluation time
            "reference": parsed["reference"],
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "document_name": chunk.document_name,
        }

    def _parse_llm_response(
        self,
        raw_text: str,
        chunk_id: str,
    ) -> dict[str, str] | None:
        """
        Parse the LLM's JSON response into a question/reference dict.

        Handles two cases:
          1. The model returned clean JSON as instructed.
          2. The model wrapped JSON in markdown code fences despite instructions.

        Args:
            raw_text: Raw string content from the LLM response.
            chunk_id: Source chunk identifier, used for error logging only.

        Returns:
            Dict with "question" and "reference" keys, or None if parsing
            failed or required fields are missing.
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

    def _append_sample(self, output_path: Path, sample: dict[str, Any]) -> None:
        """
        Append a single sample as a JSON line to the output file.

        Uses append mode so existing samples are never overwritten.
        The file is created if it does not yet exist.

        Args:
            output_path: Path to the target .jsonl file.
            sample:      Dict conforming to the JSONL schema.
        """
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")