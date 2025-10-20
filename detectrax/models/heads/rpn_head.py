"""Region Proposal Network (RPN) head implemented with Flax.

This module consumes multi-scale FPN feature maps and produces two outputs per
level:

* Objectness logits estimating whether an anchor contains an object.
* Bounding-box regression deltas relative to the anchors.

The design closely follows the RPN head described in Section 3.1 of the Faster
R-CNN paper, with a shared 3x3 convolution followed by classification and
regression branches using 1x1 convolutions. The module expects the canonical
five-level Feature Pyramid Network tensor dictionary with keys ``"P2"`` through
``"P6"`` and preserves the spatial resolution of each level.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float

FPNFeature = Float[Array, "batch height width channels"]
ObjectnessMap = Float[Array, "batch height width anchors"]
DeltaMap = Float[Array, "batch height width anchor_deltas"]


class RPNHead(nn.Module):
    """Region Proposal Network head with shared weights across FPN levels.

    The module applies a shared 3x3 convolution (256 channels) followed by two
    parallel 1x1 convolutions that predict objectness logits and bounding-box
    deltas respectively. Parameters are shared across all pyramid levels.

    Attributes:
        num_anchors: Number of anchors per spatial location (default: ``3``).
        prior_prob: Prior probability used to initialise the objectness bias.
            The bias value is set to ``-log((1 - prior_prob) / prior_prob)``.
        dtype: Computation dtype for the convolutions.
        level_names: Expected FPN level keys (defaults to ``("P2", ..., "P6")``).
    """

    num_anchors: int = 3
    prior_prob: float = 0.01
    dtype: Any = jnp.float32
    level_names: tuple[str, ...] = ("P2", "P3", "P4", "P5", "P6")

    @nn.compact
    def __call__(
        self,
        features: Mapping[str, FPNFeature],
    ) -> tuple[dict[str, ObjectnessMap], dict[str, DeltaMap]]:
        """Compute objectness logits and box deltas for every FPN level.

        Args:
            features: Mapping from FPN level names to feature tensors shaped
                ``[batch, height, width, channels]``. The mapping must include
                entries for every level listed in :attr:`level_names`.

        Returns:
            Tuple ``(objectness, deltas)`` where each element is a dictionary
            keyed by FPN level name:

            * ``objectness[level]`` has shape
              ``[batch, height, width, num_anchors]``.
            * ``deltas[level]`` has shape
              ``[batch, height, width, num_anchors * 4]``.

        Raises:
            KeyError: If any expected FPN level is missing from ``features``.
        """
        missing = [level for level in self.level_names if level not in features]
        if missing:
            raise KeyError(f"Missing FPN feature levels: {missing}")

        bias_value = -math.log((1.0 - self.prior_prob) / self.prior_prob)
        shared_conv = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
            name="shared_conv",
        )
        objectness_conv = nn.Conv(
            features=self.num_anchors,
            kernel_size=(1, 1),
            padding="SAME",
            dtype=self.dtype,
            bias_init=nn.initializers.constant(bias_value),
            name="objectness_conv",
        )
        box_delta_conv = nn.Conv(
            features=self.num_anchors * 4,
            kernel_size=(1, 1),
            padding="SAME",
            dtype=self.dtype,
            name="box_delta_conv",
        )

        def process_level(feature_map: FPNFeature) -> tuple[ObjectnessMap, DeltaMap]:
            hidden = shared_conv(feature_map)
            hidden = nn.relu(hidden)
            objectness = objectness_conv(hidden)
            deltas = box_delta_conv(hidden)
            return objectness, deltas

        objectness_outputs: dict[str, ObjectnessMap] = {}
        delta_outputs: dict[str, DeltaMap] = {}

        for level in self.level_names:
            objectness, deltas = process_level(features[level])
            objectness_outputs[level] = objectness
            delta_outputs[level] = deltas

        return objectness_outputs, delta_outputs


__all__ = ["RPNHead"]
