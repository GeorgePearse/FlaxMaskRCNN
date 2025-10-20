"""Region-of-interest heads for two-stage detectors."""

from __future__ import annotations

__all__: list[str] = []

try:  # Optional while components are implemented incrementally.
    from .base_roi_head import BaseRoIHead
except ImportError:  # pragma: no cover - module may be absent mid-development.
    BaseRoIHead = None  # type: ignore[assignment]
else:  # pragma: no cover - exercised once module is available.
    __all__.append("BaseRoIHead")

try:
    from .bbox_heads import BBoxHead
except ImportError:  # pragma: no cover - module may be absent mid-development.
    BBoxHead = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("BBoxHead")

try:
    from .mask_heads import FCNMaskHead
except ImportError:  # pragma: no cover - module may be absent mid-development.
    FCNMaskHead = None  # type: ignore[assignment]
else:  # pragma: no cover
    __all__.append("FCNMaskHead")
