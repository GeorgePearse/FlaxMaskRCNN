"""Mask prediction loss for Mask R-CNN style heads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

MaskLogits = Float[Array, "..."]  # (batch?, num_rois, num_classes, mask_h, mask_w)
MaskTargetsArray = Float[Array, "..."]  # (batch?, num_rois, mask_h, mask_w)
MaskClasses = Int[Array, "..."]  # (batch?, num_rois)
PositiveMask = Bool[Array, "..."]  # (batch?, num_rois)


def _extract_targets(mask_targets: Any) -> tuple[MaskTargetsArray, MaskClasses]:
    """Extract mask and class targets from supported container types."""
    if isinstance(mask_targets, Mapping):
        try:
            masks = mask_targets["masks"]
            classes = mask_targets["classes"]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError("mask_targets dict must contain 'masks' and 'classes' keys.") from exc
        return masks, classes

    if hasattr(mask_targets, "masks") and hasattr(mask_targets, "classes"):
        return mask_targets.masks, mask_targets.classes

    raise ValueError("mask_targets must provide 'masks' and 'classes' via mapping keys or attributes.")


def _sigmoid_binary_cross_entropy(logits: Float[Array, ...], targets: Float[Array, ...]) -> Float[Array, ...]:
    """Numerically stable sigmoid cross-entropy."""
    logits = jnp.asarray(logits, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    max_logits = jnp.maximum(logits, 0.0)
    log_exp = jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return max_logits - logits * targets + log_exp


def _normalize_positive_indices(
    positive_indices: Array,
    expected_shape: tuple[int, int],
) -> PositiveMask:
    """Convert positive indices to a boolean mask with the expected shape."""
    positive_array = jnp.asarray(positive_indices)

    if positive_array.size == 0:
        return jnp.zeros(expected_shape, dtype=jnp.bool_)

    if positive_array.dtype == jnp.bool_:
        mask = positive_array
    else:
        positive_array = positive_array.astype(jnp.int32)
        if positive_array.ndim == 1:
            if expected_shape[0] != 1:
                raise ValueError("positive_indices with shape (num_pos,) is only supported for unbatched inputs.")
            valid = positive_array >= 0
            safe_indices = jnp.clip(positive_array, 0, max(expected_shape[1] - 1, 0))
            if expected_shape[1] == 0:
                mask = jnp.zeros((1, 0), dtype=jnp.bool_)
            else:
                one_hot = jax.nn.one_hot(safe_indices, expected_shape[1], dtype=jnp.int32)
                one_hot = jnp.asarray(one_hot, dtype=jnp.bool_)
                mask_row = jnp.any(jnp.logical_and(one_hot, valid[:, None]), axis=0)
                mask = mask_row[None, :]
        elif positive_array.ndim == 2 and positive_array.shape[0] == expected_shape[0]:
            valid = positive_array >= 0
            safe_indices = jnp.clip(positive_array, 0, max(expected_shape[1] - 1, 0))
            if expected_shape[1] == 0:
                mask = jnp.zeros(expected_shape, dtype=jnp.bool_)
            else:
                one_hot = jax.nn.one_hot(safe_indices, expected_shape[1], dtype=jnp.int32)
                one_hot = jnp.asarray(one_hot, dtype=jnp.bool_)
                mask = jnp.any(jnp.logical_and(one_hot, valid[..., None]), axis=1)
        else:
            raise ValueError(
                f"positive_indices must be boolean mask of shape {expected_shape} or integer indices shaped (num_pos,) / (batch, num_pos)."
            )

    if mask.shape == expected_shape:
        return mask.astype(jnp.bool_)

    if mask.ndim == 1 and mask.shape[0] == expected_shape[1]:
        return mask[None, :].astype(jnp.bool_)

    if mask.ndim == 2 and mask.shape[0] == expected_shape[0] and mask.shape[1] == expected_shape[1]:
        return mask.astype(jnp.bool_)

    if mask.ndim == 2 and mask.shape[0] == expected_shape[1] and expected_shape[0] == 1:
        return mask[None, :].astype(jnp.bool_)

    raise ValueError(f"positive_indices shape {mask.shape} is incompatible with expected shape {expected_shape}.")


def mask_loss(
    mask_pred: MaskLogits,
    mask_targets: Any,
    positive_indices: Array,
) -> Float[Array, ""]:
    """Compute the per-pixel binary cross-entropy loss for mask predictions.

    Args:
        mask_pred: Mask logits with shape ``(batch, num_rois, num_classes, H, W)``
            or ``(num_rois, num_classes, H, W)``.
        mask_targets: Container providing ``masks`` (``(batch, num_rois, H, W)``)
            and ``classes`` (``(batch, num_rois)``). Accepts either a mapping with
            ``\"masks\"``/``\"classes\"`` keys or an object exposing ``.masks`` and
            ``.classes`` attributes.
        positive_indices: Boolean mask or integer indices identifying the proposals
            matched to ground-truth boxes. Shapes ``(batch, num_rois)``, ``(num_rois,)``,
            ``(batch, num_pos)`` or ``(num_pos,)`` are supported.

    Returns:
        Scalar mask loss averaged across positive RoIs and pixels.
    """
    logits = jnp.asarray(mask_pred, dtype=jnp.float32)
    if logits.ndim not in (4, 5):
        raise ValueError(f"mask_pred must have 4 or 5 dimensions; received shape {logits.shape}.")

    if logits.ndim == 4:
        logits = logits[None, ...]

    masks_array, classes_array = _extract_targets(mask_targets)
    masks = jnp.asarray(masks_array, dtype=jnp.float32)
    classes = jnp.asarray(classes_array, dtype=jnp.int32)

    if masks.ndim == 3:
        masks = masks[None, ...]
    if classes.ndim == 1:
        classes = classes[None, ...]

    batch_size, num_rois, num_classes, mask_height, mask_width = logits.shape
    if masks.shape[:2] != (batch_size, num_rois):
        raise ValueError(f"mask targets must match mask_pred leading dimensions; received masks shape {masks.shape} for logits shape {logits.shape}.")
    if masks.shape[-2:] != (mask_height, mask_width):
        raise ValueError(f"mask targets spatial dimensions {masks.shape[-2:]} must equal logits dimensions {(mask_height, mask_width)}.")
    if classes.shape != (batch_size, num_rois):
        raise ValueError(f"class targets must have shape {(batch_size, num_rois)}; received {classes.shape}.")

    if num_rois == 0:
        return jnp.asarray(0.0, dtype=logits.dtype)

    positive_mask = _normalize_positive_indices(positive_indices, (batch_size, num_rois))
    positive_mask = jnp.broadcast_to(positive_mask, (batch_size, num_rois))

    invalid_negative = jnp.logical_and(classes < 0, positive_mask)
    invalid_high = jnp.logical_and(classes >= num_classes, positive_mask)
    if bool(jnp.any(invalid_negative)):  # pragma: no cover - defensive branch
        raise ValueError("Positive RoIs must have non-negative class labels.")
    if bool(jnp.any(invalid_high)):  # pragma: no cover - defensive branch
        raise ValueError("Positive RoIs reference classes outside mask_pred channels.")

    safe_classes = jnp.clip(classes, 0, num_classes - 1)
    gather_indices = safe_classes[..., None, None, None]
    gathered_logits = jnp.take_along_axis(logits, gather_indices, axis=2)
    gathered_logits = jnp.squeeze(gathered_logits, axis=2)

    loss_per_pixel = _sigmoid_binary_cross_entropy(gathered_logits, masks)

    roi_mask = positive_mask[..., None, None]
    num_positive = jnp.sum(positive_mask.astype(jnp.float32))

    def no_positives() -> Float[Array, ""]:
        return jnp.asarray(0.0, dtype=loss_per_pixel.dtype)

    def positives_present() -> Float[Array, ""]:
        total_pixels = num_positive * (mask_height * mask_width)
        loss_sum = jnp.sum(loss_per_pixel * roi_mask)
        return loss_sum / total_pixels

    return jax.lax.cond(num_positive > 0, positives_present, no_positives)


__all__ = ["mask_loss"]
