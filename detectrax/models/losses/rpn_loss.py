"""Training loss for the Region Proposal Network (RPN)."""

from __future__ import annotations

import math

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

Scalar = Float[Array, ""]

_MAX_SAMPLES = 256
_POSITIVE_FRACTION = 0.5
_MAX_POSITIVES = int(_MAX_SAMPLES * _POSITIVE_FRACTION)
_SMOOTH_L1_BETA = 1.0


def _reshape_logits(logits: jnp.ndarray) -> tuple[jnp.ndarray, int]:
    """Flatten objectness predictions or targets to ``[batch, anchors]``."""
    logits = jnp.asarray(logits, dtype=jnp.float32)
    if logits.ndim == 0:
        raise ValueError("Inputs must include a batch dimension.")
    batch_size = logits.shape[0]
    return logits.reshape((batch_size, -1)), batch_size


def _reshape_targets(targets: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Flatten targets to ``[batch, anchors]`` while preserving dtype."""
    targets = jnp.asarray(targets)
    if targets.ndim == 0 or targets.shape[0] != batch_size:
        raise ValueError("Targets must share the same batch dimension as predictions.")
    return targets.reshape((batch_size, -1))


def _reshape_weights(weights: jnp.ndarray | None, batch_size: int, anchor_count: int) -> jnp.ndarray:
    """Flatten per-anchor weights to ``[batch, anchors]``."""
    if weights is None:
        return jnp.ones((batch_size, anchor_count), dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if weights.ndim == 0 or weights.shape[0] != batch_size:
        raise ValueError("Weights must share the same batch dimension as predictions.")
    flattened = weights.reshape((batch_size, -1))
    if flattened.shape[1] != anchor_count:
        raise ValueError("Weights must align with the number of anchors.")
    return flattened


def _reshape_box_deltas(deltas: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Flatten box deltas to ``[batch, anchors, 4]``."""
    deltas = jnp.asarray(deltas, dtype=jnp.float32)
    if deltas.ndim == 0 or deltas.shape[0] != batch_size:
        raise ValueError("Box deltas must share the same batch dimension as predictions.")
    per_batch_elements = math.prod(deltas.shape[1:])
    if per_batch_elements % 4 != 0:
        raise ValueError("Box deltas must contain 4 values per anchor.")
    anchor_count = per_batch_elements // 4
    return deltas.reshape((batch_size, anchor_count, 4))


def _smooth_l1_loss(pred: jnp.ndarray, target: jnp.ndarray, *, beta: float = _SMOOTH_L1_BETA) -> jnp.ndarray:
    """Element-wise Smooth L1 loss."""
    diff = jnp.abs(pred - target)
    quadratic = 0.5 * diff * diff / beta
    linear = diff - 0.5 * beta
    return jnp.where(diff < beta, quadratic, linear)


def _balanced_sample_masks(
    targets: jnp.ndarray,
    weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Select a 1:1 positive/negative subset capped at 256 anchors."""
    positive_mask = (targets == 1) & (weights > 0)
    negative_mask = (targets == 0) & (weights > 0)

    positive_int = positive_mask.astype(jnp.int32)
    negative_int = negative_mask.astype(jnp.int32)

    pos_counts = jnp.sum(positive_int, axis=1)
    neg_counts = jnp.sum(negative_int, axis=1)

    max_positives = jnp.asarray(_MAX_POSITIVES, dtype=jnp.int32)
    max_samples = jnp.asarray(_MAX_SAMPLES, dtype=jnp.int32)

    pos_limit = jnp.minimum(pos_counts, max_positives)
    remaining = jnp.maximum(max_samples - pos_limit, 0)
    neg_limit = jnp.minimum(neg_counts, remaining)

    pos_cumsum = jnp.cumsum(positive_int, axis=1)
    neg_cumsum = jnp.cumsum(negative_int, axis=1)

    pos_keep = positive_mask & (pos_cumsum <= pos_limit[:, None])
    neg_keep = negative_mask & (neg_cumsum <= neg_limit[:, None])
    sample_mask = pos_keep | neg_keep
    return pos_keep, neg_keep, sample_mask


def rpn_loss(
    objectness_pred: Float[Array, "batch ..."],
    box_deltas_pred: Float[Array, "batch ..."],
    objectness_targets: Array,
    box_delta_targets: Float[Array, "batch ..."],
    weights: Array | None = None,
) -> tuple[Scalar, Scalar, Scalar]:
    """Compute the classification and regression losses for the RPN.

    Args:
        objectness_pred: Predicted objectness logits with shape ``[batch, ...]``.
        box_deltas_pred: Predicted box deltas with shape ``[batch, ..., 4]`` after reshaping.
        objectness_targets: Ground-truth objectness targets in ``{1, 0, -1}``.
        box_delta_targets: Ground-truth box deltas aligned with ``box_deltas_pred``.
        weights: Optional per-anchor weights. Anchors with non-positive weights do not
            contribute to the loss.

    Returns:
        Tuple ``(total_loss, cls_loss, reg_loss)`` where each element is a scalar.
    """
    logits_flat, batch_size = _reshape_logits(objectness_pred)
    targets_flat = _reshape_targets(objectness_targets, batch_size)

    anchor_count = logits_flat.shape[1]
    deltas_pred_flat = _reshape_box_deltas(box_deltas_pred, batch_size)
    deltas_target_flat = _reshape_box_deltas(box_delta_targets, batch_size)

    if deltas_pred_flat.shape != deltas_target_flat.shape:
        raise ValueError("Predicted and target box deltas must share the same shape.")
    if deltas_pred_flat.shape[1] != anchor_count:
        raise ValueError("Box deltas must align with the number of objectness anchors.")

    weights_flat = _reshape_weights(weights, batch_size, anchor_count)
    weights_flat = jnp.where(targets_flat >= 0, weights_flat, 0.0)

    labels = jnp.clip(targets_flat.astype(jnp.float32), 0.0, 1.0)

    pos_keep, neg_keep, sample_mask = _balanced_sample_masks(targets_flat, weights_flat)

    sample_weights = weights_flat * sample_mask.astype(weights_flat.dtype)
    cls_per_anchor = optax.sigmoid_binary_cross_entropy(logits_flat, labels)
    cls_num = jnp.sum(cls_per_anchor * sample_weights)
    cls_den = jnp.sum(sample_weights)
    cls_loss = jnp.where(cls_den > 0.0, cls_num / cls_den, 0.0)

    positive_weights = weights_flat * pos_keep.astype(weights_flat.dtype)
    smooth_l1 = _smooth_l1_loss(deltas_pred_flat, deltas_target_flat)
    reg_per_anchor = jnp.sum(smooth_l1, axis=-1)
    reg_num = jnp.sum(reg_per_anchor * positive_weights)
    reg_den = jnp.sum(positive_weights)
    reg_loss = jnp.where(reg_den > 0.0, reg_num / reg_den, 0.0)

    total_loss = cls_loss + reg_loss
    return total_loss, cls_loss, reg_loss


__all__ = ["rpn_loss"]
