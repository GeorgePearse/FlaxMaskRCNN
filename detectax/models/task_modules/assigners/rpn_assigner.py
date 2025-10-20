"""RPN ground-truth target assignment."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from detectax.models.utils.box_coder import encode_boxes
from detectax.models.utils.iou import box_iou

_M_AXIS = "M"
_M_WITH_COORDS = "M 4"
_N_AXIS = "N"
_N_WITH_COORDS = "N 4"


def _validate_inputs(
    anchors: jnp.ndarray,
    gt_boxes: jnp.ndarray,
    gt_labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError("anchors must have shape (num_anchors, 4).")
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError("gt_boxes must have shape (num_boxes, 4).")
    if gt_boxes.shape[0] != gt_labels.shape[0]:
        raise ValueError("gt_boxes and gt_labels must contain the same number of entries.")
    return anchors, gt_boxes, gt_labels


def assign_rpn_targets(
    anchors: Float[Array, _N_WITH_COORDS],
    gt_boxes: Float[Array, _M_WITH_COORDS],
    gt_labels: Int[Array, _M_AXIS],
    *,
    pos_iou_threshold: float = 0.7,
    neg_iou_threshold: float = 0.3,
) -> tuple[Int[Array, _N_AXIS], Float[Array, _N_WITH_COORDS], Float[Array, _N_AXIS]]:
    """Assign labels, box deltas, and weights to anchors for the RPN.

    Args:
        anchors: Anchor boxes in ``(x1, y1, x2, y2)`` format.
        gt_boxes: Ground-truth boxes aligned with ``gt_labels``.
        gt_labels: Ground-truth class labels. A value of ``-1`` marks an
            ignore/crowd region. Non-negative values correspond to foreground
            classes.
        pos_iou_threshold: IoU threshold for positive anchors.
        neg_iou_threshold: IoU threshold below which anchors are negative.

    Returns:
        ``labels`` with values ``1`` (positive), ``0`` (negative), or ``-1``
        (ignored), ``target_deltas`` encoding matched GT boxes for positives,
        and ``weights`` where non-ignored anchors receive weight ``1``.

    Notes:
        The function is pure and JAX-compatible, enabling usage with
        :func:`jax.vmap` for batched processing.
    """

    anchors = jnp.asarray(anchors, dtype=jnp.float32)
    gt_boxes = jnp.asarray(gt_boxes, dtype=jnp.float32)
    gt_labels = jnp.asarray(gt_labels, dtype=jnp.int32)

    anchors, gt_boxes, gt_labels = _validate_inputs(anchors, gt_boxes, gt_labels)

    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]

    if num_gt == 0:
        labels = jnp.zeros((num_anchors,), dtype=jnp.int32)
        target_deltas = jnp.zeros((num_anchors, 4), dtype=jnp.float32)
        weights = jnp.ones((num_anchors,), dtype=jnp.float32)
        return labels, target_deltas, weights

    ious = box_iou(anchors, gt_boxes)

    valid_mask = gt_labels >= 0
    crowd_mask = gt_labels == -1

    valid_ious = jnp.where(valid_mask, ious, -jnp.inf)
    best_iou = jnp.max(jnp.where(valid_mask, ious, 0.0), axis=1)
    best_gt_idx = jnp.argmax(valid_ious, axis=1)

    positive_mask = best_iou > pos_iou_threshold
    negative_mask = (best_iou < neg_iou_threshold) & (~positive_mask)

    crowd_overlap = jnp.max(jnp.where(crowd_mask, ious, 0.0), axis=1)
    crowd_ignore = crowd_overlap > 0.0

    labels = -jnp.ones((num_anchors,), dtype=jnp.int32)
    labels = jnp.where(negative_mask & ~crowd_ignore, 0, labels)
    labels = jnp.where(positive_mask, 1, labels)
    labels = jnp.where(crowd_ignore & ~positive_mask, -1, labels)

    matched_gt_boxes = gt_boxes[best_gt_idx]
    deltas = encode_boxes(matched_gt_boxes, anchors)
    target_deltas = jnp.where(positive_mask[:, None], deltas, 0.0)

    weights = jnp.where(labels >= 0, 1.0, 0.0)

    return labels, target_deltas, weights


__all__ = ["assign_rpn_targets"]
