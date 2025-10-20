"""Bounding-box classification and regression head for Fast/Mask R-CNN."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from detectrax.models.roi_heads.base_roi_head import BaseRoIHead

RoIBatch = Float[Array, "batch rois height width channels"]
BoxFeatures = Float[Array, "batch rois hidden"]
ClassificationLogits = Float[Array, "batch rois num_classes"]
RegressionDeltas = Float[Array, "batch rois box_dim"]


class BBoxHead(BaseRoIHead):
    """Two-layer MLP head for final classification and bounding-box regression.

    The module mirrors the Fast R-CNN design: pooled RoI features are flattened
    and passed through two fully-connected layers with ReLU activations before
    branching into classification and regression predictors. The regression
    branch can operate in either class-specific or class-agnostic mode.

    Attributes:
        num_classes: Number of foreground classes to predict.
        in_channels: Channel dimension of pooled RoI features.
        hidden_dim: Width of the shared fully-connected layers.
        class_agnostic: Whether to predict a single set of box deltas per RoI.
        dtype: Computation dtype for the dense layers.
    """

    num_classes: int = 80
    in_channels: int = 256
    hidden_dim: int = 1024
    class_agnostic: bool = False
    dtype: Any = jnp.float32

    def setup(self) -> None:
        super().setup()
        kernel_init = nn.initializers.lecun_normal()
        self.fc1 = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=kernel_init,
            name="fc1",
        )
        self.fc2 = nn.Dense(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=kernel_init,
            name="fc2",
        )
        self.cls_predictor = nn.Dense(
            features=self.num_classes,
            dtype=self.dtype,
            kernel_init=kernel_init,
            name="cls_logits",
        )
        reg_outputs = 4 if self.class_agnostic else self.num_classes * 4
        self.bbox_predictor = nn.Dense(
            features=reg_outputs,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.001),
            name="bbox_deltas",
        )

    # ------------------------------------------------------------------
    # Utility helpers --------------------------------------------------
    # ------------------------------------------------------------------
    def _flatten_roi_features(self, roi_features: RoIBatch) -> Float[Array, "batch rois flat"]:
        """Flatten pooled RoI features whilst validating their dimensions."""
        if roi_features.ndim != 5:
            raise ValueError("RoI features must have shape [batch, rois, height, width, channels].")

        height, width = self.roi_output_size
        if roi_features.shape[-3] != height or roi_features.shape[-2] != width:
            raise ValueError(
                f"RoI feature spatial dimensions mismatch: expected {height}x{width}, got {roi_features.shape[-3]}x{roi_features.shape[-2]}"
            )

        if roi_features.shape[-1] != self.in_channels:
            raise ValueError(f"RoI feature channel dimension mismatch: expected {self.in_channels}, got {roi_features.shape[-1]}")

        leading = roi_features.shape[:2]
        flat_dim = height * width * self.in_channels
        return roi_features.reshape(leading + (flat_dim,))

    # ------------------------------------------------------------------
    # BaseRoIHead API --------------------------------------------------
    # ------------------------------------------------------------------
    def box_head(self, roi_features: RoIBatch, train: bool = False) -> BoxFeatures:
        del train  # Unused but kept for API compatibility.
        hidden = self._flatten_roi_features(roi_features)
        hidden = nn.relu(self.fc1(hidden))
        hidden = nn.relu(self.fc2(hidden))
        return hidden

    def get_predictions(self, box_features: BoxFeatures) -> dict[str, jax.Array]:
        cls_logits = self.cls_predictor(box_features)
        bbox_deltas = self.bbox_predictor(box_features)
        return {"cls_logits": cls_logits, "bbox_deltas": bbox_deltas}

    def get_targets(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Target generation is handled by dedicated assigners.")

    def loss(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Loss computation is delegated to training utilities.")

    # ------------------------------------------------------------------
    # Flax module execution --------------------------------------------
    # ------------------------------------------------------------------
    def __call__(
        self,
        roi_features: RoIBatch,
        *,
        train: bool = False,
    ) -> tuple[ClassificationLogits, RegressionDeltas]:
        hidden = self.box_head(roi_features, train=train)
        predictions = self.get_predictions(hidden)
        return predictions["cls_logits"], predictions["bbox_deltas"]


__all__ = ["BBoxHead"]
