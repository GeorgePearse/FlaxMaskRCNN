"""Training losses for the Fast R-CNN detection head."""

from __future__ import annotations

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

__all__ = ["detection_loss"]

LossTuple = tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]

_MAX_SAMPLES_PER_IMAGE = 512
_MAX_POSITIVE_FRACTION = 0.25  # 1 : 3 ratio
_EPSILON = 1e-8


def _ensure_class_dimension(
    box_deltas: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    """Ensure box deltas are shaped [B, N, num_classes, 4]."""
    if box_deltas.ndim == 3:
        if box_deltas.shape[-1] != num_classes * 4:
            raise ValueError(f"Expected last dimension {num_classes * 4} for flattened box deltas, got {box_deltas.shape[-1]}.")
        return box_deltas.reshape(box_deltas.shape[0], box_deltas.shape[1], num_classes, 4)

    if box_deltas.ndim == 4 and box_deltas.shape[-1] == 4:
        if box_deltas.shape[-2] != num_classes:
            raise ValueError(f"Expected class dimension size {num_classes}, got {box_deltas.shape[-2]}.")
        return box_deltas

    raise ValueError("box_deltas must be shaped [batch, num_rois, num_classes * 4] or [batch, num_rois, num_classes, 4].")


def _smooth_l1(diff: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """Smooth L1 loss applied element-wise."""
    abs_diff = jnp.abs(diff)
    if beta < _EPSILON:
        return abs_diff
    return jnp.where(abs_diff < beta, 0.5 * abs_diff * abs_diff / beta, abs_diff - 0.5 * beta)


def _gather_class_specific(values: jnp.ndarray, class_indices: jnp.ndarray) -> jnp.ndarray:
    """Select class-specific slices for each ROI."""
    batch, num_rois, num_classes, coords = values.shape
    if coords != 4:
        raise ValueError("Box deltas must have 4 coordinates per class.")
    roi_indices = jnp.arange(num_rois, dtype=jnp.int32)

    def gather_single(image_values: jnp.ndarray, image_classes: jnp.ndarray) -> jnp.ndarray:
        return image_values[roi_indices, image_classes, :]

    return jax.vmap(gather_single, in_axes=(0, 0), out_axes=0)(values, class_indices)


def _prepare_targets(
    box_delta_targets: jnp.ndarray,
    num_classes: int,
    class_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Normalise target tensor to [B, N, 4] class-specific layout."""
    if box_delta_targets.ndim == 3 and box_delta_targets.shape[-1] == 4:
        return box_delta_targets

    if box_delta_targets.ndim == 3 and box_delta_targets.shape[-1] == num_classes * 4:
        reshaped = box_delta_targets.reshape(
            box_delta_targets.shape[0],
            box_delta_targets.shape[1],
            num_classes,
            4,
        )
        return _gather_class_specific(reshaped, class_indices)

    if box_delta_targets.ndim == 4 and box_delta_targets.shape[-2:] == (num_classes, 4):
        return _gather_class_specific(box_delta_targets, class_indices)

    raise ValueError(
        "box_delta_targets must be shaped [batch, num_rois, 4], [batch, num_rois, num_classes * 4], or [batch, num_rois, num_classes, 4]."
    )


def detection_loss(
    cls_scores: Float[Array, "batch num_rois num_classes_plus_bg"],
    box_deltas: Float[Array, ...],
    cls_targets: Int[Array, "batch num_rois"],
    box_delta_targets: Float[Array, ...],
    weights: Float[Array, "batch num_rois"] | None = None,
) -> LossTuple:
    """Compute Fast R-CNN detection head losses.

    Args:
        cls_scores: Classification logits shaped ``[batch, num_rois, num_classes + 1]``.
        box_deltas: Bounding-box deltas shaped ``[batch, num_rois, num_classes * 4]`` or
            ``[batch, num_rois, num_classes, 4]``.
        cls_targets: Integer class targets ``[batch, num_rois]`` with ``0`` representing background
            and ``1..num_classes`` the foreground classes. Entries ``< 0`` are ignored.
        box_delta_targets: Regression targets with shapes matching ``box_deltas`` or ``[batch, num_rois, 4]``.
        weights: Optional per-sample weights shaped ``[batch, num_rois]``. Values ``<= 0`` mask samples.

    Returns:
        Tuple ``(total_loss, classification_loss, regression_loss)``.
    """
    cls_scores = jnp.asarray(cls_scores, dtype=jnp.float32)
    if cls_scores.ndim != 3:
        raise ValueError("cls_scores must be 3-D with shape [batch, num_rois, num_classes + 1].")

    batch_size, num_rois, num_classes_plus_bg = cls_scores.shape
    if num_classes_plus_bg < 2:
        raise ValueError("cls_scores must contain at least one foreground class and background.")
    num_classes = num_classes_plus_bg - 1

    cls_targets = jnp.asarray(cls_targets, dtype=jnp.int32)
    if cls_targets.shape != (batch_size, num_rois):
        raise ValueError("cls_targets must be shaped [batch, num_rois].")

    if weights is None:
        weights_arr = jnp.ones((batch_size, num_rois), dtype=jnp.float32)
    else:
        weights_arr = jnp.asarray(weights, dtype=jnp.float32)
        if weights_arr.shape != (batch_size, num_rois):
            raise ValueError("weights must be shaped [batch, num_rois].")

    box_deltas = jnp.asarray(box_deltas, dtype=jnp.float32)
    if box_deltas.shape[0] != batch_size or box_deltas.shape[1] != num_rois:
        raise ValueError("box_deltas must align with the batch and RoI dimensions.")

    box_deltas = _ensure_class_dimension(box_deltas, num_classes)

    box_delta_targets = jnp.asarray(box_delta_targets, dtype=jnp.float32)
    if box_delta_targets.shape[0] != batch_size or box_delta_targets.shape[1] != num_rois:
        raise ValueError("box_delta_targets must align with the batch and RoI dimensions.")

    valid_mask = (weights_arr > 0) & (cls_targets >= 0)
    if bool(jnp.any((cls_targets > num_classes) & valid_mask)):
        raise ValueError("cls_targets contain class indices outside the valid range.")

    positive_mask = (cls_targets > 0) & valid_mask
    negative_mask = (cls_targets == 0) & valid_mask

    max_pos = int(_MAX_SAMPLES_PER_IMAGE * _MAX_POSITIVE_FRACTION)
    desired_pos = jnp.minimum(jnp.sum(positive_mask, axis=1), max_pos)
    desired_neg = jnp.minimum(
        jnp.sum(negative_mask, axis=1),
        _MAX_SAMPLES_PER_IMAGE - desired_pos,
    )

    pos_rank = jnp.cumsum(positive_mask.astype(jnp.int32), axis=1) - 1
    neg_rank = jnp.cumsum(negative_mask.astype(jnp.int32), axis=1) - 1
    sel_pos = positive_mask & (pos_rank < desired_pos[:, None])
    sel_neg = negative_mask & (neg_rank < desired_neg[:, None])
    selected_mask = sel_pos | sel_neg

    sample_weights = weights_arr * selected_mask.astype(weights_arr.dtype)
    cls_targets_safe = jnp.where(valid_mask, cls_targets, 0)

    log_probs = jnn.log_softmax(cls_scores, axis=-1)
    per_roi_ce = -jnp.squeeze(
        jnp.take_along_axis(log_probs, cls_targets_safe[..., None], axis=-1),
        axis=-1,
    )
    per_roi_ce = per_roi_ce * sample_weights

    cls_normalizer = jnp.maximum(jnp.sum(sample_weights), _EPSILON).astype(cls_scores.dtype)
    cls_loss = jnp.sum(per_roi_ce) / cls_normalizer

    positive_selected = sel_pos
    reg_weights = weights_arr * positive_selected.astype(weights_arr.dtype)
    class_indices = jnp.clip(cls_targets - 1, 0, num_classes - 1).astype(jnp.int32)

    pred_deltas = _gather_class_specific(box_deltas, class_indices)
    target_deltas = _prepare_targets(box_delta_targets, num_classes, class_indices)

    diff = pred_deltas - target_deltas
    per_coord_loss = _smooth_l1(diff)
    per_roi_reg = jnp.sum(per_coord_loss, axis=-1) * reg_weights

    reg_normalizer = jnp.maximum(jnp.sum(reg_weights), _EPSILON).astype(cls_scores.dtype)
    reg_loss = jnp.where(reg_normalizer > _EPSILON, jnp.sum(per_roi_reg) / reg_normalizer, 0.0)

    total_loss = cls_loss + reg_loss
    return total_loss.astype(cls_scores.dtype), cls_loss.astype(cls_scores.dtype), reg_loss.astype(cls_scores.dtype)
