"""Non-maximum suppression utilities.

This module implements a JAX-friendly non-maximum suppression (NMS)
routine that mirrors the traditional algorithm used in two-stage detection
pipelines. The implementation is purely functional, relies on JAX control
flow primitives, and supports batched inputs through :func:`jax.vmap`.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .iou import box_iou

SingleBoxes = Float[Array, "num_boxes 4"]
BatchedBoxes = Float[Array, "batch num_boxes 4"]
SingleScores = Float[Array, "num_boxes"]
BatchedScores = Float[Array, "batch num_boxes"]
SingleIndices = Int[Array, "max_kept"]
BatchedIndices = Int[Array, "batch max_kept"]
CountScalar = Int[Array, ""]
BatchedCounts = Int[Array, "batch"]

_NEGATIVE_ONE: Final[int] = -1


class NMSResult(NamedTuple):
    """Container for NMS outputs.

    Attributes:
        indices: Indices of the selected boxes in descending score order with
            a fixed shape. Unused slots are filled with ``-1``.
        valid_counts: Number of valid indices per example. Values can be used
            to slice ``indices`` or construct masks.
    """

    indices: SingleIndices | BatchedIndices
    valid_counts: CountScalar | BatchedCounts


def _nms_single(
    boxes: SingleBoxes,
    scores: SingleScores,
    iou_threshold: float,
    max_output_size: int,
) -> tuple[SingleIndices, CountScalar]:
    """Run NMS on a single set of boxes."""
    boxes = jnp.asarray(boxes, dtype=jnp.float32)
    scores = jnp.asarray(scores, dtype=jnp.float32)

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (N, 4); received {boxes.shape}.")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError(f"scores must have shape (N,); received {scores.shape} for {boxes.shape[0]} boxes.")
    if max_output_size < 0:
        raise ValueError(f"max_output_size must be non-negative; received {max_output_size}.")

    num_boxes = boxes.shape[0]
    if num_boxes == 0 or max_output_size == 0:
        return jnp.zeros((0,), dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32)

    max_kept = min(num_boxes, max_output_size)
    _, sorted_indices = jax.lax.top_k(scores, num_boxes)

    suppressed = jnp.zeros((num_boxes,), dtype=jnp.bool_)
    kept_indices = jnp.full((max_kept,), _NEGATIVE_ONE, dtype=jnp.int32)

    def cond_fn(state: tuple[Int[Array, ""], Int[Array, ""], Array, Array]) -> Array:
        i, kept_count, _, _ = state
        return jnp.logical_and(i < num_boxes, kept_count < max_kept)

    def body_fn(state: tuple[Int[Array, ""], Int[Array, ""], Array, Array]) -> tuple[Int[Array, ""], Int[Array, ""], Array, Array]:
        i, kept_count, suppressed_mask, kept = state
        current_index = jax.lax.dynamic_index_in_dim(sorted_indices, i, axis=0, keepdims=False)

        def skip_fn(
            operand: tuple[Int[Array, ""], Int[Array, ""], Array, Array, Int[Array, ""]],
        ) -> tuple[Int[Array, ""], Int[Array, ""], Array, Array]:
            idx_i, count, mask, selected, _ = operand
            return idx_i + 1, count, mask, selected

        def select_fn(
            operand: tuple[Int[Array, ""], Int[Array, ""], Array, Array, Int[Array, ""]],
        ) -> tuple[Int[Array, ""], Int[Array, ""], Array, Array]:
            idx_i, count, mask, selected, candidate = operand
            candidate_box = jax.lax.dynamic_slice(boxes, (candidate, 0), (1, 4))
            ious = box_iou(candidate_box, boxes)[0]
            new_mask = jnp.logical_or(mask, ious > iou_threshold)
            new_mask = new_mask.at[candidate].set(True)
            new_selected = selected.at[count].set(candidate)
            return idx_i + 1, count + 1, new_mask, new_selected

        current_suppressed = jax.lax.dynamic_index_in_dim(suppressed_mask, current_index, axis=0, keepdims=False)
        return jax.lax.cond(
            current_suppressed,
            skip_fn,
            select_fn,
            (i, kept_count, suppressed_mask, kept, current_index),
        )

    initial_state = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        suppressed,
        kept_indices,
    )
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    _, valid_count, _, final_indices = final_state

    mask = jnp.arange(max_kept, dtype=jnp.int32) < valid_count
    final_indices = jnp.where(mask, final_indices, _NEGATIVE_ONE)
    return final_indices, valid_count


def nms(
    boxes: SingleBoxes | BatchedBoxes,
    scores: SingleScores | BatchedScores,
    iou_threshold: float = 0.7,
    max_output_size: int = 1000,
) -> NMSResult:
    """Perform (batched) non-maximum suppression.

    Args:
        boxes: Bounding boxes in ``(x1, y1, x2, y2)`` format. Accepts either a
            single set of boxes with shape ``(N, 4)`` or a batched tensor
            shaped ``(B, N, 4)``.
        scores: Confidence scores associated with ``boxes``. Must match the
            leading dimensions of ``boxes`` (``(N,)`` or ``(B, N)``).
        iou_threshold: IoU overlap threshold used to suppress boxes. Boxes with
            ``IoU > threshold`` are removed.
        max_output_size: Maximum number of boxes to retain per example.

    Returns:
        An :class:`NMSResult` containing the selected box indices (sorted by
        score) and the number of valid indices. Unused entries are filled with
        ``-1``; they can be ignored using the accompanying ``valid_counts``.
    """
    boxes_array = jnp.asarray(boxes, dtype=jnp.float32)
    scores_array = jnp.asarray(scores, dtype=jnp.float32)

    if boxes_array.ndim == 2:
        indices, valid_count = _nms_single(boxes_array, scores_array, iou_threshold, max_output_size)
        return NMSResult(indices=indices, valid_counts=valid_count)

    if boxes_array.ndim != 3 or boxes_array.shape[-1] != 4:
        raise ValueError(f"boxes must have shape (N, 4) or (B, N, 4); received {boxes_array.shape}.")
    if scores_array.ndim != 2 or scores_array.shape != boxes_array.shape[:2]:
        raise ValueError(f"scores must have shape (N,) or (B, N) matching boxes; received {scores_array.shape} for boxes {boxes_array.shape}.")

    single_nms = jax.vmap(_nms_single, in_axes=(0, 0, None, None))
    indices, counts = single_nms(boxes_array, scores_array, iou_threshold, max_output_size)

    max_kept = indices.shape[-1] if indices.ndim == 2 else 0
    if max_kept > 0:
        valid_mask = jnp.arange(max_kept, dtype=jnp.int32)
        valid_mask = valid_mask[None, :] < counts[:, None]
        indices = jnp.where(valid_mask, indices, _NEGATIVE_ONE)

    return NMSResult(indices=indices, valid_counts=counts)


__all__ = ["nms", "NMSResult"]
