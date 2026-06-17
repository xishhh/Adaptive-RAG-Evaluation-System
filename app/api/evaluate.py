"""
app/api/evaluate.py

Evaluation endpoints — Phase 6 implementation.

POST /evaluate  — runs the RAGAS evaluation harness against a JSONL dataset.
GET  /metrics   — returns historical evaluation metrics and aggregate summaries.

Architecture:
  POST /evaluate:
    EvaluateRequest (dataset_path, run_label)
        → RagasEvaluator.run(dataset_path)   [loads JSONL, runs RAGAS]
        → MetricsTracker.save_run(...)        [persists report to disk]
        → EvaluationResponse

  GET /metrics:
    MetricsTracker.list_runs()               [reads evaluation_results/]
    MetricsTracker.get_aggregate_metrics()   [computes means across runs]
        → MetricsSummaryResponse

Design decisions:
  1. RagasEvaluator and MetricsTracker are module-level singletons.
     They are stateless between requests, but constructing ChatOpenAI
     on every request wastes time and risks rate-limit spikes.
  2. dataset_path is treated as a filesystem path relative to the
     working directory. Absolute paths also work.
  3. Evaluation is synchronous. For large datasets this is slow, but
     async job queuing is out of scope for Phase 6.
  4. HTTPException codes:
       404 — dataset file not found
       422 — dataset is malformed (missing fields, invalid JSON)
       500 — unexpected RAGAS or I/O failure
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_metrics_tracker, get_ragas_evaluator
from app.evaluators.metrics_tracker import MetricsTracker
from app.evaluators.ragas_evaluator import RagasEvaluator
from app.models.requests import EvaluateRequest
from app.models.responses import (
    EvaluationResponse,
    EvaluationRunRecord,
    MetricsSummaryResponse,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()




@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Run Evaluation",
    description=(
        "Run the RAGAS evaluation harness against a JSONL evaluation dataset. "
        "Each line of the dataset must contain: question, answer, contexts (list), reference."
    ),
    tags=["Evaluation"],
    status_code=status.HTTP_200_OK,
)
async def run_evaluation(
    request: EvaluateRequest,
    ragas_evaluator: RagasEvaluator = Depends(get_ragas_evaluator),
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker),
) -> EvaluationResponse:
    """
    Evaluate system quality using RAGAS metrics.

    The dataset at `dataset_path` must be a JSONL file where each line is:
        {
          "question":  "<user question>",
          "answer":    "<rag-generated answer>",
          "contexts":  ["<chunk text 1>", "<chunk text 2>", ...],
          "reference": "<ground-truth answer>"
        }
    """
    dataset_path = Path(request.dataset_path)
    logger.info(
        "POST /evaluate | label='%s' | dataset='%s'",
        request.run_label,
        dataset_path,
    )

    # ------------------------------------------------------------------ #
    # Run evaluation                                                        #
    # ------------------------------------------------------------------ #
    try:
        # Run in a thread so RAGAS gets a clean event loop (avoids conflict with FastAPI's loop).
        scores = await asyncio.to_thread(ragas_evaluator.run, dataset_path)
    except FileNotFoundError as exc:
        logger.warning("Evaluation dataset not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.warning("Malformed evaluation dataset: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during RAGAS evaluation.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------ #
    # Persist run report                                                    #
    # ------------------------------------------------------------------ #
    # Load dataset once more just to count samples (already validated above).
    try:
        dataset = ragas_evaluator.load_dataset(dataset_path)
        sample_count = len(dataset)
    except Exception:
        sample_count = 0

    run_id = metrics_tracker.save_run(
        run_label=request.run_label,
        dataset_path=str(dataset_path),
        scores=scores,
        sample_count=sample_count,
    )
    created_at = datetime.now(tz=timezone.utc).isoformat()

    logger.info(
        "Evaluation run '%s' complete | scores=%s",
        run_id,
        scores,
    )

    return EvaluationResponse(
        run_id=run_id,
        run_label=request.run_label,
        dataset_path=str(dataset_path),
        sample_count=sample_count,
        created_at=created_at,
        metrics=scores,
        message=f"Evaluation run '{request.run_label}' completed successfully.",
    )


@router.get(
    "/metrics",
    response_model=MetricsSummaryResponse,
    summary="Get Evaluation Metrics",
    description=(
        "Retrieve historical evaluation runs and aggregate metric summaries. "
        "Returns up to 50 most recent runs, newest first."
    ),
    tags=["Evaluation"],
    status_code=status.HTTP_200_OK,
)
async def get_metrics(
    metrics_tracker: MetricsTracker = Depends(get_metrics_tracker),
) -> MetricsSummaryResponse:
    """
    Return all historical evaluation runs and per-metric aggregate scores.
    """
    logger.info("GET /metrics called.")

    runs_raw = metrics_tracker.list_runs(limit=50)
    aggregate = metrics_tracker.get_aggregate_metrics()

    runs = [EvaluationRunRecord(**record) for record in runs_raw]

    return MetricsSummaryResponse(
        total_runs=len(runs_raw),
        aggregate_metrics=aggregate,
        runs=runs,
    )