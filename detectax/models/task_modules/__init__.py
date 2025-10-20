"""Task-specific modules for detection models."""

from __future__ import annotations

__all__: list[str] = []

try:  # Optional until all assigners are implemented.
    from .assigners import assign_detection_targets, assign_rpn_targets
except Exception:  # pragma: no cover - dependencies may be missing during early development.
    assign_detection_targets = None  # type: ignore[assignment]
    assign_rpn_targets = None  # type: ignore[assignment]
else:
    __all__.extend(["assign_detection_targets", "assign_rpn_targets"])

try:
    from .mask_target_generator import generate_mask_targets
except Exception:  # pragma: no cover - mask utilities remain optional.
    generate_mask_targets = None  # type: ignore[assignment]
else:
    __all__.append("generate_mask_targets")

try:
    from .proposal_generator import generate_proposals
except Exception:  # pragma: no cover - proposal generator may be absent in some builds.
    generate_proposals = None  # type: ignore[assignment]
else:
    __all__.append("generate_proposals")

try:
    from .post_processors import postprocess_detections
except Exception:  # pragma: no cover - post-processing not always available.
    postprocess_detections = None  # type: ignore[assignment]
else:
    __all__.append("postprocess_detections")
