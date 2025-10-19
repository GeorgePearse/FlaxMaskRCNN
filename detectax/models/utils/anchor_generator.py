"""Anchor box generator for Feature Pyramid Networks (FPN).

This module implements the standard anchor generation algorithm used by
Faster/Mask R-CNN style detectors. For each spatial location of a feature
map, multiple anchors are placed with different scales and aspect ratios.
Anchors are defined in absolute `(x1, y1, x2, y2)` pixel coordinates, making
them directly consumable by downstream Region Proposal Networks (RPNs).

Key features:
    * Supports configurable strides, base sizes, aspect ratios, and scales.
    * Generates anchors for multiple pyramid levels (default: P2–P6).
    * Purely functional implementation using :func:`jax.vmap` for efficiency.
    * Provides both functional and object-oriented APIs.

The implementation follows the PyTorch reference in
``reference/visdet_models/task_modules/prior_generators/anchor_generator.py``,
while adopting idiomatic JAX/NumPy semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

LevelShapeMapping = Mapping[str, tuple[int, int]]


def _as_float_array(values: Sequence[float]) -> jnp.ndarray:
    """Convert a float sequence to a ``float32`` JAX array."""
    return jnp.asarray(values, dtype=jnp.float32)


@dataclass(frozen=True, kw_only=True)
class AnchorGenerator:
    """Generator for multi-scale, multi-aspect-ratio anchor boxes.

    Attributes:
        strides: Pixel strides for each FPN level (e.g., ``[4, 8, 16, 32, 64]``).
        base_sizes: Base anchor sizes per level (same length as ``strides``).
        aspect_ratios: Ratios of height/width for anchors.
        scales: Multiplicative scales applied on top of ``base_sizes``.
        level_names: Human-readable names for each level (default ``P2``–``P6``).

    Notes:
        * The number of generated anchors per spatial location equals
          ``len(aspect_ratios) * len(scales)``.
        * Anchor centers are aligned with feature grid centers via
          ``(index + 0.5) * stride``.
    """

    strides: Sequence[float] = (4.0, 8.0, 16.0, 32.0, 64.0)
    base_sizes: Sequence[float] = (32.0, 64.0, 128.0, 256.0, 512.0)
    aspect_ratios: Sequence[float] = (0.5, 1.0, 2.0)
    scales: Sequence[float] = (1.0,)
    level_names: Sequence[str] = ("P2", "P3", "P4", "P5", "P6")

    def __post_init__(self) -> None:
        """Validate configuration and cache arrays for computation."""
        num_levels = len(self.strides)
        if not (len(self.base_sizes) == len(self.level_names) == num_levels):
            raise ValueError(
                "strides, base_sizes, and level_names must share the same length; "
                f"got {len(self.strides)}, {len(self.base_sizes)}, "
                f"and {len(self.level_names)}."
            )
        if len(self.aspect_ratios) == 0:
            raise ValueError("aspect_ratios must contain at least one value.")
        if len(self.scales) == 0:
            raise ValueError("scales must contain at least one value.")
        if any(r <= 0 for r in self.aspect_ratios):
            raise ValueError(f"aspect_ratios must be positive; received {self.aspect_ratios}.")
        if any(s <= 0 for s in self.scales):
            raise ValueError(f"scales must be positive; received {self.scales}.")
        if any(s <= 0 for s in self.base_sizes):
            raise ValueError(f"base_sizes must be positive; received {self.base_sizes}.")
        if any(s <= 0 for s in self.strides):
            raise ValueError(f"strides must be positive; received {self.strides}.")

        object.__setattr__(self, "_ratios", _as_float_array(self.aspect_ratios))
        object.__setattr__(self, "_scales", _as_float_array(self.scales))
        object.__setattr__(self, "_base_sizes", _as_float_array(self.base_sizes))
        object.__setattr__(self, "_strides", _as_float_array(self.strides))

    @property
    def anchors_per_location(self) -> int:
        """Return the number of anchors generated per feature location."""
        return len(self.aspect_ratios) * len(self.scales)

    def generate(
        self,
        feature_map_shapes: Sequence[tuple[int, int]] | LevelShapeMapping,
        *,
        per_level: bool = False,
    ) -> dict[str, Float[Array, "num_anchors 4"]] | Float[Array, "total_anchors 4"]:
        """Generate anchors for all configured pyramid levels.

        Args:
            feature_map_shapes: Either a sequence of ``(height, width)`` tuples
                ordered according to :attr:`level_names`, or a mapping from
                level name to ``(height, width)``.
            per_level: If ``True``, return a dictionary keyed by level name.

        Returns:
            Anchors stacked across levels when ``per_level=False``; otherwise a
            dictionary mapping level name to its anchors.
        """
        normalized_shapes = self._normalize_shapes(feature_map_shapes)
        anchors_by_level = {}
        for (level, (height, width)), stride, base_size in zip(normalized_shapes, self._strides, self._base_sizes):
            anchors_by_level[level] = self._generate_level_anchors(height, width, stride, base_size)

        if per_level:
            return anchors_by_level

        arrays = tuple(anchors_by_level[level] for level, _ in normalized_shapes)
        return jnp.concatenate(arrays, axis=0) if arrays else jnp.zeros((0, 4), dtype=jnp.float32)

    def _generate_level_anchors(
        self,
        height: int,
        width: int,
        stride: float,
        base_size: float,
    ) -> Float[Array, "num_anchors 4"]:
        """Generate anchors for a single feature level."""
        if height < 0 or width < 0:
            raise ValueError(f"Feature map shape must be non-negative, got ({height}, {width}).")
        if height == 0 or width == 0:
            return jnp.zeros((0, 4), dtype=jnp.float32)

        ratios_sqrt = jnp.sqrt(self._ratios)
        widths = base_size * (self._scales[:, None] / ratios_sqrt[None, :])
        heights = base_size * (self._scales[:, None] * ratios_sqrt[None, :])

        widths = widths.reshape(-1)
        heights = heights.reshape(-1)
        half_widths = 0.5 * widths
        half_heights = 0.5 * heights

        base_boxes = jnp.stack(
            (-half_widths, -half_heights, half_widths, half_heights),
            axis=-1,
        )

        grid_x = (jnp.arange(width, dtype=jnp.float32) + 0.5) * stride
        grid_y = (jnp.arange(height, dtype=jnp.float32) + 0.5) * stride
        centers_x, centers_y = jnp.meshgrid(grid_x, grid_y, indexing="xy")
        centers = jnp.stack((centers_x.reshape(-1), centers_y.reshape(-1)), axis=-1)

        def shift_base(center: Float[Array, 2]) -> Float[Array, "anchors 4"]:
            center_xyxy = jnp.tile(center, 2)
            return base_boxes + center_xyxy

        anchors = jax.vmap(shift_base)(centers)
        return anchors.reshape(-1, 4)

    def _normalize_shapes(
        self,
        feature_map_shapes: Sequence[tuple[int, int]] | LevelShapeMapping,
    ) -> tuple[tuple[str, tuple[int, int]], ...]:
        """Standardize feature map shapes to align with configured levels."""
        if isinstance(feature_map_shapes, Mapping):
            missing = [lvl for lvl in self.level_names if lvl not in feature_map_shapes]
            if missing:
                raise KeyError(f"Missing feature map shapes for levels: {missing}")
            ordered = tuple((level, feature_map_shapes[level]) for level in self.level_names)
        else:
            if len(feature_map_shapes) != len(self.level_names):
                raise ValueError(
                    f"feature_map_shapes must match number of pyramid levels; expected {len(self.level_names)}, got {len(feature_map_shapes)}."
                )
            ordered = tuple(zip(self.level_names, feature_map_shapes))

        for level, (height, width) in ordered:
            if height < 0 or width < 0:
                raise ValueError(f"Negative spatial dimension for level {level}: {(height, width)}")
        return ordered


def generate_pyramid_anchors(
    feature_map_shapes: Sequence[tuple[int, int]] | LevelShapeMapping,
    *,
    strides: Sequence[float] = (4.0, 8.0, 16.0, 32.0, 64.0),
    base_sizes: Sequence[float] = (32.0, 64.0, 128.0, 256.0, 512.0),
    aspect_ratios: Sequence[float] = (0.5, 1.0, 2.0),
    scales: Sequence[float] = (1.0,),
    level_names: Sequence[str] = ("P2", "P3", "P4", "P5", "P6"),
) -> Float[Array, "total_anchors 4"]:
    """Functional API for anchor generation.

    Args:
        feature_map_shapes: Spatial shapes for each pyramid level.
        strides: Pixel strides per level.
        base_sizes: Base sizes (nominal edge lengths) per level.
        aspect_ratios: Anchor aspect ratios (height / width).
        scales: Per-anchor multiplicative scales.
        level_names: Names for each pyramid level.

    Returns:
        All anchors flattened across levels with shape ``[N, 4]``.
    """
    generator = AnchorGenerator(
        strides=strides,
        base_sizes=base_sizes,
        aspect_ratios=aspect_ratios,
        scales=scales,
        level_names=level_names,
    )
    return generator.generate(feature_map_shapes, per_level=False)
