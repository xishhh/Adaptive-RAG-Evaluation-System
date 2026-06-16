"""
app/evaluators/ragas_evaluator.py

RAGAS evaluation engine for Phase 6.

Responsibilities:
  - Load a JSONL evaluation dataset from disk.
  - Construct a RAGAS EvaluationDataset.
  - Run all four required metrics: context_precision, context_recall,
    faithfulness, answer_relevancy.
  - Return a flat dict of metric name → score.

Design decisions:
  1. RAGAS LLM and embeddings are constructed from the project's own
     Settings so they route through OpenRouter (OPENAI_API_BASE), exactly
     as the rest of the system does. RAGAS must never auto-discover its
     own keys independently.
  2. This class has no file-writing responsibility. It receives a Path,
     runs evaluation, and returns scores. MetricsTracker owns persistence.
  3. The dataset JSONL schema requires four fields per sample:
       - question:  the user question (str)
       - answer:    the RAG-generated answer (str)
       - contexts:  list of retrieved chunk texts (list[str])
       - reference: the ground-truth answer (str)
     context_recall and context_precision require `reference`.
     faithfulness and answer_relevancy do not, but including it is
     harmless and makes the dataset format consistent.
  4. RAGAS v0.2+ uses EvaluationDataset + ragas.evaluate(). The older
     Dataset.from_dict() approach is not used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RagasEvaluator:
    """
    Runs RAGAS evaluation over a JSONL dataset file.

    Each line of the JSONL file must be a JSON object with these fields:
        question  (str)        — the user question
        answer    (str)        — the RAG-generated answer
        contexts  (list[str])  — retrieved chunk texts used for the answer
        reference (str)        — the ground-truth / expected answer

    Usage:
        evaluator = RagasEvaluator()
        scores = evaluator.evaluate(Path("data/evaluation_dataset/sample_eval.jsonl"))
        # → {"context_precision": 0.85, "context_recall": 0.78, ...}
    """

    def __init__(self) -> None:
        settings = get_settings()

        # Re-use the same LLM config as the rest of the system.
        llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            temperature=0.0,
        )
        embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
        )

        # Wrap for RAGAS.
        self._llm = LangchainLLMWrapper(llm)
        self._embeddings = LangchainEmbeddingsWrapper(embeddings)

        self._metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]

        logger.info(
            "RagasEvaluator initialised | model=%s | embedding=%s | metrics=%s",
            settings.LLM_MODEL,
            settings.EMBEDDING_MODEL,
            [m.name for m in self._metrics],
        )

    def load_dataset(self, dataset_path: Path) -> Dataset:
        """
        Load and validate a JSONL evaluation dataset.

        Args:
            dataset_path: Absolute or relative path to the .jsonl file.

        Returns:
            HuggingFace Dataset with columns: question, answer, contexts, reference.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty or a required field is missing.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        required_fields = {"question", "answer", "contexts", "reference"}
        samples: list[dict[str, Any]] = []

        with dataset_path.open("r", encoding="utf-8") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {dataset_path}: {exc}"
                    ) from exc

                missing = required_fields - sample.keys()
                if missing:
                    raise ValueError(
                        f"Line {line_number} is missing required fields: {missing}"
                    )

                # Normalise: contexts must be a list of strings.
                if isinstance(sample["contexts"], str):
                    sample["contexts"] = [sample["contexts"]]

                samples.append(
                    {
                        "question": sample["question"],
                        "answer": sample["answer"],
                        "contexts": sample["contexts"],
                        "reference": sample["reference"],
                    }
                )

        if not samples:
            raise ValueError(f"Evaluation dataset is empty: {dataset_path}")

        logger.info(
            "Loaded %d evaluation samples from '%s'.", len(samples), dataset_path
        )
        return Dataset.from_list(samples)

    def run(self, dataset_path: Path) -> dict[str, float | None]:
        """
        Run all RAGAS metrics over the dataset at dataset_path.

        Args:
            dataset_path: Path to the JSONL evaluation dataset file.

        Returns:
            Dict mapping metric name to float score (0.0–1.0).
            A metric score is None if RAGAS failed to compute it.

        Raises:
            FileNotFoundError: Propagated from load_dataset.
            ValueError:        Propagated from load_dataset.
        """
        dataset = self.load_dataset(dataset_path)

        logger.info(
            "Running RAGAS evaluation | dataset=%s | samples=%d | metrics=%s",
            dataset_path.name,
            len(dataset),
            [m.name for m in self._metrics],
        )

        result = evaluate(
            dataset=dataset,
            metrics=self._metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            raise_exceptions=False,   # return NaN rather than aborting on one bad sample
        )

        # result is a ragas.result.Result; .scores is a Dataset of per-sample dicts.
        # We aggregate to mean per metric.
        scores: dict[str, float | None] = {}
        result_df = result.to_pandas()

        for metric in self._metrics:
            col = metric.name
            if col in result_df.columns:
                mean_val = result_df[col].dropna().mean()
                scores[col] = float(mean_val) if not __import__("math").isnan(mean_val) else None
            else:
                logger.warning("Metric '%s' not found in RAGAS result columns.", col)
                scores[col] = None

        logger.info("RAGAS evaluation complete | scores=%s", scores)
        return scores