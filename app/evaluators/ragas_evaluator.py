from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value
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
    def __init__(self) -> None:
        settings = get_settings()

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
                    raise ValueError(f"Invalid JSON on line {line_number} of {dataset_path}: {exc}") from exc

                missing = required_fields - sample.keys()
                if missing:
                    raise ValueError(f"Line {line_number} is missing required fields: {missing}")

                if isinstance(sample["contexts"], str):
                    sample["contexts"] = [sample["contexts"]]

                samples.append({
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "contexts": sample["contexts"],
                    "ground_truth": sample["reference"],
                })

        if not samples:
            raise ValueError(f"Evaluation dataset is empty: {dataset_path}")

        logger.info("Loaded %d evaluation samples from '%s'.", len(samples), dataset_path)
        features = Features({
            "question": Value("string"),
            "answer": Value("string"),
            "contexts": Sequence(Value("string")),
            "ground_truth": Value("string"),
        })
        return Dataset.from_list(samples, features=features)

    def run(self, dataset_path: Path) -> dict[str, float | None]:
        dataset = self.load_dataset(dataset_path)

        logger.info("Running RAGAS evaluation | dataset=%s | samples=%d | metrics=%s", dataset_path.name, len(dataset), [m.name for m in self._metrics])

        result = evaluate(
            dataset=dataset,
            metrics=self._metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            raise_exceptions=False,
        )

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
