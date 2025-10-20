"""Base RoI head implementation for two-stage detectors."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from detectrax.models.layers.roi_align import roi_align

_ROIS_AXIS = "rois"
_ROIS_WITH_COORDS = "rois 4"
_ROIS_WITH_FEATURES = "rois out_h out_w channels"


class BaseRoIHead(nn.Module, ABC):
    """Base class encapsulating RoI feature extraction for FPN inputs."""

    roi_output_size: tuple[int, int] = (7, 7)
    sampling_ratio: int = 2
    canonical_scale: float = 224.0
    canonical_level: int = 4
    feature_strides: dict[str, int] = field(
        default_factory=lambda: {
            "p2": 4,
            "p3": 8,
            "p4": 16,
            "p5": 32,
            "p6": 64,
        }
    )

    def setup(self) -> None:
        if not self.feature_strides:
            raise ValueError("feature_strides must be non-empty for RoI extraction")

        sorted_levels = sorted(self.feature_strides.items(), key=lambda item: item[1])

        for level, stride in sorted_levels:
            if stride <= 0:
                raise ValueError(f"Stride for level '{level}' must be positive, got {stride}")

        self._level_names = tuple(level for level, _ in sorted_levels)
        self._level_numbers = {level: int(round(math.log2(stride))) for level, stride in sorted_levels}
        self._spatial_scales = {level: 1.0 / float(stride) for level, stride in sorted_levels}

        self._number_to_level = {num: level for level, num in self._level_numbers.items()}
        self._min_level_num = min(self._level_numbers.values())
        self._max_level_num = max(self._level_numbers.values())

    # ------------------------------------------------------------------
    # Abstract interface ------------------------------------------------
    # ------------------------------------------------------------------
    @abstractmethod
    def box_head(
        self,
        roi_features: Float[Array, "batch rois height width channels"],
        train: bool = False,
    ) -> Float[Array, "batch rois hidden"]:
        """Apply the task-specific box head."""

    @abstractmethod
    def get_targets(
        self,
        proposals: Float[Array, "batch rois 4"],
        gt_boxes: Float[Array, "batch max_gt 4"],
        gt_labels: Int[Array, "batch max_gt"],
    ) -> dict:
        """Generate training targets for the RoI head."""

    @abstractmethod
    def get_predictions(
        self,
        box_features: Float[Array, "batch rois hidden"],
    ) -> dict:
        """Convert head outputs into detection predictions."""

    @abstractmethod
    def loss(self, *args, **kwargs) -> dict:
        """Compute RoI head losses. Signature delegated to subclasses."""

    # ------------------------------------------------------------------
    # Shared utilities --------------------------------------------------
    # ------------------------------------------------------------------
    def extract_roi_features(
        self,
        features: Mapping[str, Float[Array, "batch height width channels"]],
        proposals: Float[Array, "batch rois 4"],
        *,
        output_size: tuple[int, int] | None = None,
        sampling_ratio: int | None = None,
    ) -> Float[Array, "batch rois out_h out_w channels"]:
        """Extract RoI-aligned features from multi-scale FPN inputs."""

        if not isinstance(features, Mapping):
            raise TypeError("features must be provided as a mapping")

        if proposals.ndim != 3 or proposals.shape[-1] != 4:
            raise ValueError("proposals must have shape [batch, num_rois, 4] in (x1, y1, x2, y2) format")

        output_size = output_size or self.roi_output_size
        sampling_ratio = sampling_ratio if sampling_ratio is not None else self.sampling_ratio

        missing_levels = [level for level in self._level_names if level not in features]
        if missing_levels:
            raise KeyError(f"Missing FPN levels required for RoI extraction: {missing_levels}")

        reference_level = self._level_names[0]
        feat_dtype = features[reference_level].dtype

        batch_size = proposals.shape[0]
        num_rois = proposals.shape[1]

        for level in self._level_names:
            feat = features[level]
            if feat.shape[0] != batch_size:
                raise ValueError(f"Feature map at level '{level}' has batch size {feat.shape[0]}, expected {batch_size}")

        channels = features[reference_level].shape[-1]

        def _assign_levels(boxes: Float[Array, _ROIS_WITH_COORDS]) -> Int[Array, _ROIS_AXIS]:
            widths = jnp.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
            heights = jnp.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
            roi_scale = jnp.sqrt(widths * heights)
            target = self.canonical_level + jnp.log2(roi_scale / self.canonical_scale)
            target = jnp.floor(target)
            target = jnp.clip(target, self._min_level_num, self._max_level_num)
            return target.astype(jnp.int32)

        batch_roi_features: list[Float[Array, _ROIS_WITH_FEATURES]] = []

        for batch_idx in range(batch_size):
            boxes = proposals[batch_idx]

            if num_rois == 0:
                empty = jnp.zeros((0, *output_size, channels), dtype=feat_dtype)
                batch_roi_features.append(empty)
                continue

            level_indices = _assign_levels(boxes)
            roi_output = jnp.zeros((num_rois, *output_size, channels), dtype=feat_dtype)

            unique_levels = jnp.unique(level_indices).tolist()

            for lvl in unique_levels:
                level_num = int(lvl)
                level_name = self._number_to_level.get(level_num)
                if level_name is None:
                    raise KeyError(f"No FPN level registered for pyramid level {level_num}")

                roi_ids = jnp.where(level_indices == level_num)[0]
                if roi_ids.size == 0:
                    continue

                feat_map = features[level_name][batch_idx : batch_idx + 1]
                selected_boxes = boxes[roi_ids]

                aligned = roi_align(
                    feat_map,
                    selected_boxes,
                    output_size=output_size,
                    spatial_scale=self._spatial_scales[level_name],
                    sampling_ratio=sampling_ratio,
                )
                roi_output = roi_output.at[roi_ids].set(aligned)

            batch_roi_features.append(roi_output)

        return jnp.stack(batch_roi_features, axis=0)


__all__ = ["BaseRoIHead"]
