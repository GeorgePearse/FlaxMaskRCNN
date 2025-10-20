"""Detection target assignment utilities for the second-stage head."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from detectax.models.utils.box_coder import encode_boxes
from detectax.models.utils.iou import box_iou

ProposalArray = Float[Array, "num_proposals 4"]
GTBoxArray = Float[Array, "num_gt 4"]
ClassIndexArray = Int[Array, "num_gt"]
OneHotLabelArray = Float[Array, "num_gt num_classes"]
LabelArray = ClassIndexArray | OneHotLabelArray
LabelOutput = Int[Array, "num_proposals"]
TargetDeltaOutput = Float[Array, "num_proposals num_classes 4"]
WeightOutput = Float[Array, "num_proposals num_classes 4"]


def _prepare_label_data(gt_labels: jnp.ndarray) -> tuple[ClassIndexArray, int]:
    """Normalize ground-truth labels and determine the number of classes.

    Args:
        gt_labels: Either integer class indices shaped ``(num_gt,)`` or a
            one-hot encoding shaped ``(num_gt, num_classes)``. Background
            should use the index ``0``.

    Returns:
        Tuple ``(class_indices, num_classes)`` with integer labels and the
        total number of classes (including background).

    Raises:
        ValueError: If ``gt_labels`` has an unsupported rank or lacks a class
            dimension when empty.
        TypeError: When attempting to infer the number of classes from traced
            values (e.g. within ``jax.vmap``) using 1D integer labels.
    """

    if gt_labels.ndim == 1:
        class_indices = jnp.asarray(gt_labels, dtype=jnp.int32)
        num_gt = class_indices.shape[0]
        if num_gt == 0:
            raise ValueError(
                "Cannot infer the number of classes from empty 1D gt_labels. Provide one-hot encoded labels to supply an explicit class dimension."
            )
        max_label = jnp.max(class_indices)
        try:
            num_classes = int(max_label) + 1
        except TypeError as exc:  # Triggered when tracing with vmap/jit.
            raise TypeError("Failed to infer num_classes from traced integer labels. Use one-hot encoded labels for JAX transformations.") from exc
        return class_indices, num_classes

    if gt_labels.ndim == 2:
        num_classes = gt_labels.shape[1]
        if num_classes < 1:
            raise ValueError("gt_labels must encode at least one class dimension.")
        if gt_labels.shape[0] == 0:
            class_indices = jnp.zeros((0,), dtype=jnp.int32)
        else:
            class_indices = jnp.argmax(gt_labels, axis=1).astype(jnp.int32)
        return class_indices, num_classes

    raise ValueError("gt_labels must be rank-1 (class indices) or rank-2 (one-hot encoded).")


def assign_detection_targets(
    proposals: ProposalArray,
    gt_boxes: GTBoxArray,
    gt_labels: LabelArray,
    *,
    pos_iou_threshold: float = 0.5,
    neg_iou_threshold: float = 0.5,
) -> tuple[LabelOutput, TargetDeltaOutput, WeightOutput]:
    """Assign classification labels and box regression targets to proposals.

    Args:
        proposals: Proposal boxes shaped ``(num_proposals, 4)`` in ``(x1, y1, x2, y2)`` format.
        gt_boxes: Ground-truth boxes aligned with ``gt_labels``.
        gt_labels: Class annotations associated with ``gt_boxes``. Accepts
            either integer class indices ``(num_gt,)`` or one-hot encodings
            ``(num_gt, num_classes)``. Background should be encoded as class ``0``.
        pos_iou_threshold: IoU threshold to mark proposals as positives.
        neg_iou_threshold: IoU threshold below which proposals become negatives.

    Returns:
        Tuple ``(labels, target_deltas, weights)`` where:

        * ``labels`` contains class indices for each proposal (background ``0``).
        * ``target_deltas`` stores class-specific regression targets with shape
          ``(num_proposals, num_classes, 4)``.
        * ``weights`` provides the same shape as ``target_deltas`` and marks
          valid regression targets with ``1``.

    Raises:
        ValueError: If inputs have incompatible shapes or invalid ranks.
    """

    proposals = jnp.asarray(proposals, dtype=jnp.float32)
    gt_boxes = jnp.asarray(gt_boxes, dtype=jnp.float32)
    gt_labels = jnp.asarray(gt_labels)

    if proposals.ndim != 2 or proposals.shape[1] != 4:
        raise ValueError(f"proposals must have shape (num_proposals, 4); received {proposals.shape}.")
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError(f"gt_boxes must have shape (num_gt, 4); received {gt_boxes.shape}.")
    if gt_labels.shape[0] != gt_boxes.shape[0]:
        raise ValueError(
            f"gt_labels and gt_boxes must reference the same number of instances; received {gt_labels.shape[0]} and {gt_boxes.shape[0]}."
        )

    class_indices, num_classes = _prepare_label_data(gt_labels)

    num_proposals = proposals.shape[0]
    output_shape = (num_proposals, num_classes, 4)

    if gt_boxes.shape[0] == 0:
        labels = jnp.zeros((num_proposals,), dtype=jnp.int32)
        target_deltas = jnp.zeros(output_shape, dtype=jnp.float32)
        weights = jnp.zeros(output_shape, dtype=jnp.float32)
        return labels, target_deltas, weights

    ious = box_iou(proposals, gt_boxes)
    best_iou = jnp.max(ious, axis=1)
    best_gt_idx = jnp.argmax(ious, axis=1)

    matched_boxes = gt_boxes[best_gt_idx]
    matched_labels = class_indices[best_gt_idx]

    positive_mask = best_iou >= pos_iou_threshold
    negative_mask = best_iou < neg_iou_threshold
    neutral_mask = jnp.logical_not(jnp.logical_or(positive_mask, negative_mask))

    labels = jnp.zeros((num_proposals,), dtype=jnp.int32)
    labels = jnp.where(positive_mask, matched_labels, labels)
    labels = jnp.where(neutral_mask, 0, labels)

    deltas = encode_boxes(matched_boxes, proposals)
    deltas = deltas * positive_mask[:, None]

    class_one_hot = jax.nn.one_hot(labels, num_classes, dtype=jnp.float32)
    regression_mask = class_one_hot * positive_mask[:, None]

    target_deltas = regression_mask[:, :, None] * deltas[:, None, :]
    weights = jnp.broadcast_to(regression_mask[:, :, None], target_deltas.shape)

    return labels, target_deltas, weights


__all__ = ["assign_detection_targets"]
