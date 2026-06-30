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
    def __init__(self) -> None:
        settings = get_settings()
        self._results_dir = Path(settings.EVALUATION_RESULTS_DIR)
        self._results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MetricsTracker initialised | results_dir=%s", self._results_dir)

    def save_run(
        self,
        run_label: str,
        dataset_path: str,
        scores: dict[str, float | None],
        sample_count: int,
    ) -> str:
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

        logger.info("Saved evaluation run '%s' → %s | metrics=%s", run_id, report_path, scores)
        return run_id

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
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
        runs = self.list_runs(limit=1000)
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

    @staticmethod
    def _generate_run_id(label: str) -> str:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", label)[:40]
        return f"{timestamp}_{slug}"
