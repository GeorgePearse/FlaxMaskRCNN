"""COCO evaluator utilities.

This module provides a thin wrapper around :mod:`pycocotools.cocoeval` so that
Detectax components can compute standard COCO metrics (bbox / mask mAP).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:  # pragma: no cover - surfaced via tests
    raise ImportError("pycocotools is required for COCO evaluation. Install it before using detectax.evaluation.coco_evaluator.") from exc


_DEFAULT_IOU_THRESHOLDS = tuple(np.linspace(0.5, 0.95, 10))


@dataclass(frozen=True)
class CocoEvaluationResult:
    """Container for the common COCO summary statistics."""

    AP: float
    AP50: float
    AP75: float
    APs: float
    APm: float
    APl: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain dictionary."""

        return {
            "AP": self.AP,
            "AP50": self.AP50,
            "AP75": self.AP75,
            "APs": self.APs,
            "APm": self.APm,
            "APl": self.APl,
        }


class COCOEvaluator:
    """Evaluate predictions using the official COCO metrics.

    Parameters
    ----------
    annotation_file:
        Path to the ground-truth COCO annotations JSON file.
    iou_thresholds:
        Iterable of IoU thresholds to evaluate. Defaults to the standard
        ``0.50:0.05:0.95`` range used by the COCO leaderboard.
    use_cats:
        Whether to use category-specific evaluation. Defaults to ``True``.
    iou_types:
        Iterable of IoU types to evaluate (``"bbox"`` and/or ``"segm"``).
    """

    def __init__(
        self,
        annotation_file: str,
        *,
        iou_thresholds: Iterable[float] | None = None,
        use_cats: bool = True,
        iou_types: Sequence[str] = ("bbox", "segm"),
    ) -> None:
        self._coco_gt = COCO(annotation_file)
        self._use_cats = use_cats
        if iou_thresholds is not None:
            thresholds = tuple(float(t) for t in sorted(iou_thresholds))
        else:
            thresholds = _DEFAULT_IOU_THRESHOLDS
        self._iou_thresholds = thresholds
        if not self._iou_thresholds:
            raise ValueError("At least one IoU threshold is required for evaluation.")
        self._iou_types = tuple(iou_types)
        if not self._iou_types:
            raise ValueError("At least one IoU type (bbox/segm) must be specified.")

    def evaluate(
        self,
        predictions: Mapping[str, Sequence[MutableMapping[str, object]]],
        *,
        verbose: bool = False,
    ) -> dict[str, CocoEvaluationResult]:
        """Run COCO evaluation for each requested IoU type.

        Parameters
        ----------
        predictions:
            Mapping from IoU type (``"bbox"`` or ``"segm"``) to an iterable of
            prediction dictionaries in COCO result format.
        verbose:
            When ``True`` the underlying :class:`COCOeval` summary is printed.

        Returns
        -------
        Dict[str, CocoEvaluationResult]
            Evaluation statistics for each IoU type.
        """

        metrics: dict[str, CocoEvaluationResult] = {}
        for iou_type in self._iou_types:
            if iou_type not in predictions:
                raise KeyError(f"Missing predictions for IoU type '{iou_type}'.")

            predictions_list = list(predictions[iou_type])
            coco_dt = self._coco_gt.loadRes(predictions_list)
            coco_eval = COCOeval(self._coco_gt, coco_dt, iou_type)
            coco_eval.params.useCats = int(self._use_cats)
            coco_eval.params.iouThrs = np.array(self._iou_thresholds, dtype=np.float64)

            if not verbose:
                # Suppress the stdout noise produced by COCOeval.summarize().
                import contextlib
                import io

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    self._run_eval(coco_eval)
            else:
                self._run_eval(coco_eval)

            metrics[iou_type] = CocoEvaluationResult(*(float(x) for x in coco_eval.stats[:6]))

        return metrics

    @staticmethod
    def _run_eval(coco_eval: COCOeval) -> None:
        """Execute the three evaluation phases."""

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


__all__ = ["COCOEvaluator", "CocoEvaluationResult"]
