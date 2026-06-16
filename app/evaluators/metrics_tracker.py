"""
app/evaluators/metrics_tracker.py

Persists evaluation run reports and provides historical metric summaries.

Responsibilities:
  - Write a timestamped JSON report for each evaluation run.
  - List all historical runs.
  - Compute aggregate (mean) metrics across all runs.

Design decisions:
  1. Storage is the local filesystem under EVALUATION_RESULTS_DIR.
     Each run is a self-contained JSON file named {run_id}.json.
     This keeps the implementation dependency-free and easily inspectable.
  2. run_id format: YYYYMMDDTHHmmss_{label_slug}.
     Sortable, human-readable, and unique at single-process scale.
  3. This class has no knowledge of RAGAS. It accepts a plain dict of
     scores from RagasEvaluator and treats them as opaque float values.
  4. Aggregate metrics are computed lazily on each GET /metrics call by
     reading all run files. Acceptable for Phase 6 volumes; a database
     would be used in production at scale.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """
    Reads and writes evaluation run reports on the local filesystem.

    Usage:
        tracker = MetricsTracker()
        run_id = tracker.save_run(
            run_label="contract_review",
            dataset_path="data/evaluation_dataset/sample_eval.jsonl",
            scores={"faithfulness": 0.91, "answer_relevancy": 0.87, ...},
            sample_count=10,
        )
        history = tracker.list_runs()
        aggregate = tracker.get_aggregate_metrics()
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._results_dir = Path(settings.EVALUATION_RESULTS_DIR)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MetricsTracker initialised | results_dir=%s", self._results_dir
        )

    # ------------------------------------------------------------------ #
    # Writing                                                              #
    # ------------------------------------------------------------------ #

    def save_run(
        self,
        run_label: str,
        dataset_path: str,
        scores: dict[str, float | None],
        sample_count: int,
    ) -> str:
        """
        Persist a completed evaluation run to disk.

        Args:
            run_label:    Human-readable label supplied by the caller.
            dataset_path: Path of the JSONL dataset that was evaluated.
            scores:       Metric name → score mapping from RagasEvaluator.
            sample_count: Number of dataset samples that were evaluated.

        Returns:
            The generated run_id string (also used as the filename stem).
        """
        run_id = self._generate_run_id(run_label)
        created_at = datetime.now(tz=timezone.utc).isoformat()

        record: dict[str, Any] = {
            "run_id": run_id,
            "run_label": run_label,
            "dataset_path": dataset_path,
            "sample_count": sample_count,
            "created_at": created_at,
            "metrics": scores,
        }

        report_path = self._results_dir / f"{run_id}.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)

        logger.info(
            "Saved evaluation run '%s' → %s | metrics=%s",
            run_id,
            report_path,
            scores,
        )
        return run_id

    # ------------------------------------------------------------------ #
    # Reading                                                              #
    # ------------------------------------------------------------------ #

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Return the most recent evaluation run records.

        Args:
            limit: Maximum number of runs to return (newest first).

        Returns:
            List of run record dicts, sorted newest-first.
        """
        run_files = sorted(
            self._results_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        records: list[dict[str, Any]] = []
        for path in run_files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    records.append(json.load(fh))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read run file '%s': %s", path, exc)

        return records

    def get_aggregate_metrics(self) -> dict[str, float | None]:
        """
        Compute mean scores for each metric across all historical runs.

        Returns:
            Dict mapping metric name → mean float score.
            A metric is None if no runs recorded a valid score for it.
        """
        runs = self.list_runs(limit=1000)  # all runs
        if not runs:
            return {}

        accumulator: dict[str, list[float]] = {}
        for run in runs:
            for metric_name, score in run.get("metrics", {}).items():
                if score is not None:
                    accumulator.setdefault(metric_name, []).append(score)

        return {
            metric: (sum(vals) / len(vals)) if vals else None
            for metric, vals in accumulator.items()
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single run record by run_id.

        Args:
            run_id: The run identifier returned by save_run().

        Returns:
            The run record dict, or None if not found.
        """
        report_path = self._results_dir / f"{run_id}.json"
        if not report_path.exists():
            return None
        try:
            with report_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Could not read run file '%s': %s", report_path, exc)
            return None

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_run_id(label: str) -> str:
        """
        Generate a sortable, filesystem-safe run identifier.

        Format: YYYYMMDDTHHmmss_{label_slug}
        Example: 20240615T143022_contract_review
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", label)[:40]
        return f"{timestamp}_{slug}"