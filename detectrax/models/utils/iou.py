"""Intersection-over-Union utilities for bounding boxes.

This module provides vectorized IoU computations that are compatible with JAX
transformations. Both :func:`box_iou` and :func:`giou` operate on sets of
bounding boxes in ``(x1, y1, x2, y2)`` format and return pairwise metrics.
"""

from __future__ import annotations

from typing import Any, Final

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

Boxes = Float[Array, "num_boxes 4"]
IoUMatrix = Float[Array, "num_boxes1 num_boxes2"]

_EPSILON: Final[float] = 1e-6


def _box_area(box: Float[Array, 4]) -> Float[Array, ""]:
    """Compute the non-negative area of a single box."""
    width = jnp.maximum(0.0, box[2] - box[0])
    height = jnp.maximum(0.0, box[3] - box[1])
    return width * height


def _pairwise_geometry(
    box1: Float[Array, 4],
    area1: Float[Array, ""],
    box2: Float[Array, 4],
    area2: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Return intersection area, union area, and enclosure area for a box pair."""
    x1 = jnp.maximum(box1[0], box2[0])
    y1 = jnp.maximum(box1[1], box2[1])
    x2 = jnp.minimum(box1[2], box2[2])
    y2 = jnp.minimum(box1[3], box2[3])

    intersection_w = jnp.maximum(0.0, x2 - x1)
    intersection_h = jnp.maximum(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    union = area1 + area2 - intersection

    enc_x1 = jnp.minimum(box1[0], box2[0])
    enc_y1 = jnp.minimum(box1[1], box2[1])
    enc_x2 = jnp.maximum(box1[2], box2[2])
    enc_y2 = jnp.maximum(box1[3], box2[3])
    enclosure_w = jnp.maximum(0.0, enc_x2 - enc_x1)
    enclosure_h = jnp.maximum(0.0, enc_y2 - enc_y1)
    enclosure = enclosure_w * enclosure_h

    return intersection, union, enclosure


def _validate_boxes(name: str, boxes: jnp.ndarray) -> Boxes:
    """Validate box tensor shape."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (N, 4); received {boxes.shape}.")
    return boxes


def box_iou(boxes1: Boxes, boxes2: Boxes) -> IoUMatrix:
    """Compute the pairwise Intersection-over-Union between two box sets."""
    boxes1 = _validate_boxes("boxes1", jnp.asarray(boxes1, dtype=jnp.float32))
    boxes2 = _validate_boxes("boxes2", jnp.asarray(boxes2, dtype=jnp.float32))

    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return jnp.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=jnp.float32)

    areas1 = jax.vmap(_box_area)(boxes1)
    areas2 = jax.vmap(_box_area)(boxes2)

    def pairwise(box1: Float[Array, 4], area1: Float[Array, ""]) -> Any:
        def iou_with(box2: Float[Array, 4], area2: Float[Array, ""]) -> Float[Array, ""]:
            intersection, union, _ = _pairwise_geometry(box1, area1, box2, area2)
            union_safe = jnp.maximum(union, 0.0) + _EPSILON
            return jnp.where(union > 0.0, intersection / union_safe, 0.0)

        return jax.vmap(iou_with, in_axes=(0, 0))(boxes2, areas2)

    return jax.vmap(pairwise, in_axes=(0, 0))(boxes1, areas1)


def giou(boxes1: Boxes, boxes2: Boxes) -> IoUMatrix:
    """Compute the pairwise Generalized IoU between two box sets."""
    boxes1 = _validate_boxes("boxes1", jnp.asarray(boxes1, dtype=jnp.float32))
    boxes2 = _validate_boxes("boxes2", jnp.asarray(boxes2, dtype=jnp.float32))

    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return jnp.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=jnp.float32)

    areas1 = jax.vmap(_box_area)(boxes1)
    areas2 = jax.vmap(_box_area)(boxes2)

    def pairwise(box1: Float[Array, 4], area1: Float[Array, ""]) -> Any:
        def giou_with(box2: Float[Array, 4], area2: Float[Array, ""]) -> Float[Array, ""]:
            intersection, union, enclosure = _pairwise_geometry(box1, area1, box2, area2)
            union_safe = jnp.maximum(union, 0.0) + _EPSILON
            base_iou = jnp.where(union > 0.0, intersection / union_safe, 0.0)

            enclosure_safe = jnp.maximum(enclosure, 0.0) + _EPSILON
            giou_term = (enclosure - union) / enclosure_safe
            value = base_iou - giou_term
            return jnp.where(enclosure > 0.0, value, base_iou)

        return jax.vmap(giou_with, in_axes=(0, 0))(boxes2, areas2)

    return jax.vmap(pairwise, in_axes=(0, 0))(boxes1, areas1)


__all__ = ["box_iou", "giou"]
