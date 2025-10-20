"""DetectAX model component exports."""

from __future__ import annotations

__all__: list[str] = []

try:  # Optional due to staged bring-up of subpackages.
    from . import task_modules  # noqa: F401
except Exception:  # pragma: no cover - subpackage may be absent mid-development.
    task_modules = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised when subpackage import succeeds.
    __all__.append("task_modules")

try:
    from . import roi_heads  # noqa: F401
except Exception:  # pragma: no cover - subpackage may be absent mid-development.
    roi_heads = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("roi_heads")

try:
    from .roi_heads import BBoxHead, FCNMaskHead  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - heads may not be implemented yet.
    BBoxHead = None  # type: ignore[assignment]
    FCNMaskHead = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.extend(["BBoxHead", "FCNMaskHead"])

try:
    from .losses import mask_loss  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - losses optional during early stages.
    mask_loss = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("mask_loss")
