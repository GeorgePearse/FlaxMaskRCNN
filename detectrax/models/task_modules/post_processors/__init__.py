"""Post-processing utilities for task-specific outputs."""

from __future__ import annotations

from .detection_postprocessor import postprocess_detections

__all__ = ["postprocess_detections"]

try:  # Optional until mask post-processing is implemented.
    from .mask_postprocessor import postprocess_masks
except ImportError:  # pragma: no cover - module not yet available.
    postprocess_masks = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised once implemented.
    __all__.append("postprocess_masks")
