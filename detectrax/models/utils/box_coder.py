"""Bounding box encoding and decoding utilities.

This module implements the parameterization introduced in Faster R-CNN
(Ren et al., 2015) for regressing bounding box coordinates relative to a set
of reference anchors. Two functional helpers are provided:

* :func:`encode_boxes` converts ground-truth boxes into normalized deltas.
* :func:`decode_boxes` applies predicted deltas to anchors and returns boxes.

Both functions operate on arrays shaped ``(..., 4)`` using the ``(x1, y1, x2, y2)``
coordinate convention and are compatible with :func:`jax.vmap` for batched
operation across arbitrary leading dimensions.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

Boxes = Float[Array, "... 4"]
Deltas = Float[Array, "... 4"]

_DEFAULT_CLIP_VALUE = 4.135
_EPSILON = 1e-7


def _validate_weights(weights: Sequence[float]) -> jnp.ndarray:
    """Convert and validate regression weights."""
    weight_array = jnp.asarray(weights, dtype=jnp.float32)
    if weight_array.shape != (4,):
        raise ValueError(f"weights must contain four values, received shape {weight_array.shape}.")
    return weight_array


def _split_boxes(boxes: Boxes) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split boxes into coordinate components."""
    return boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]


def encode_boxes(
    boxes: Boxes,
    anchors: Boxes,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    *,
    epsilon: float = _EPSILON,
) -> Deltas:
    """Encode ground-truth boxes relative to anchor boxes.

    Args:
        boxes: Ground-truth boxes in ``(x1, y1, x2, y2)`` format with arbitrary
            leading batch dimensions.
        anchors: Anchor boxes sharing the same shape as ``boxes``.
        weights: Normalization factors ``(wx, wy, ww, wh)`` applied to the
            resulting deltas.
        epsilon: Minimum value used to avoid divide-by-zero in width/height.

    Returns:
        Normalized deltas ``(dx, dy, dw, dh)`` with the same leading shape
        as the inputs.

    Notes:
        All computations follow the Faster R-CNN parameterization:

        ``dx = (box_cx - anchor_cx) / anchor_w / wx``

        ``dy = (box_cy - anchor_cy) / anchor_h / wy``

        ``dw = log(box_w / anchor_w) / ww``

        ``dh = log(box_h / anchor_h) / wh``
    """
    boxes = jnp.asarray(boxes, dtype=jnp.float32)
    anchors = jnp.asarray(anchors, dtype=jnp.float32)
    if boxes.shape != anchors.shape:
        raise ValueError(f"boxes and anchors must share the same shape; got {boxes.shape} and {anchors.shape}.")

    weights_array = _validate_weights(weights)
    wx, wy, ww, wh = weights_array

    ax1, ay1, ax2, ay2 = _split_boxes(anchors)
    bx1, by1, bx2, by2 = _split_boxes(boxes)

    anchor_widths = ax2 - ax1
    anchor_heights = ay2 - ay1
    box_widths = bx2 - bx1
    box_heights = by2 - by1

    anchor_widths_safe = jnp.maximum(anchor_widths, epsilon)
    anchor_heights_safe = jnp.maximum(anchor_heights, epsilon)
    box_widths_safe = jnp.maximum(box_widths, epsilon)
    box_heights_safe = jnp.maximum(box_heights, epsilon)

    anchor_cx = ax1 + 0.5 * anchor_widths
    anchor_cy = ay1 + 0.5 * anchor_heights
    box_cx = bx1 + 0.5 * box_widths
    box_cy = by1 + 0.5 * box_heights

    dx = (box_cx - anchor_cx) / anchor_widths_safe / wx
    dy = (box_cy - anchor_cy) / anchor_heights_safe / wy
    dw = jnp.log(box_widths_safe / anchor_widths_safe) / ww
    dh = jnp.log(box_heights_safe / anchor_heights_safe) / wh

    return jnp.stack((dx, dy, dw, dh), axis=-1)


def decode_boxes(
    deltas: Deltas,
    anchors: Boxes,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    *,
    clip_value: float = _DEFAULT_CLIP_VALUE,
    epsilon: float = _EPSILON,
) -> Boxes:
    """Decode predicted deltas back into pixel-aligned bounding boxes.

    Args:
        deltas: Box regression deltas in ``(dx, dy, dw, dh)`` format.
        anchors: Anchor boxes that correspond to ``deltas``.
        weights: Normalization factors ``(wx, wy, ww, wh)`` that were applied
            during encoding.
        clip_value: Maximum absolute value allowed for the scaled ``dw`` and
            ``dh`` terms before exponentiation. The default value of ``4.135``
            corresponds to the common Detectron/Mask R-CNN setting and prevents
            excessively large boxes during decoding.
        epsilon: Minimum width/height used to avoid numerical issues.

    Returns:
        Decoded boxes in ``(x1, y1, x2, y2)`` format with the same leading
        dimensions as the inputs.
    """
    deltas = jnp.asarray(deltas, dtype=jnp.float32)
    anchors = jnp.asarray(anchors, dtype=jnp.float32)
    if deltas.shape != anchors.shape:
        raise ValueError(f"deltas and anchors must share the same shape; got {deltas.shape} and {anchors.shape}.")

    weights_array = _validate_weights(weights)
    wx, wy, ww, wh = weights_array

    ax1, ay1, ax2, ay2 = _split_boxes(anchors)
    dx, dy, dw, dh = _split_boxes(deltas)

    anchor_widths = ax2 - ax1
    anchor_heights = ay2 - ay1
    anchor_cx = ax1 + 0.5 * anchor_widths
    anchor_cy = ay1 + 0.5 * anchor_heights

    anchor_widths_safe = jnp.maximum(anchor_widths, epsilon)
    anchor_heights_safe = jnp.maximum(anchor_heights, epsilon)

    scaled_dx = dx * wx
    scaled_dy = dy * wy
    scaled_dw = jnp.clip(dw * ww, a_min=-clip_value, a_max=clip_value)
    scaled_dh = jnp.clip(dh * wh, a_min=-clip_value, a_max=clip_value)

    pred_cx = scaled_dx * anchor_widths_safe + anchor_cx
    pred_cy = scaled_dy * anchor_heights_safe + anchor_cy
    pred_w = jnp.exp(scaled_dw) * anchor_widths_safe
    pred_h = jnp.exp(scaled_dh) * anchor_heights_safe

    x1 = pred_cx - 0.5 * pred_w
    y1 = pred_cy - 0.5 * pred_h
    x2 = pred_cx + 0.5 * pred_w
    y2 = pred_cy + 0.5 * pred_h

    return jnp.stack((x1, y1, x2, y2), axis=-1)


__all__ = ["encode_boxes", "decode_boxes"]
