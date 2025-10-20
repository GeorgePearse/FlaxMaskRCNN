"""Mask post-processing utilities.

This module converts per-RoI mask logits produced by the mask head into
binary instance masks aligned with the original image resolution. The
implementation closely mirrors the post-processing described in Mask R-CNN:

1. Select the predicted mask corresponding to each detected class.
2. Apply a sigmoid activation to transform logits into probabilities.
3. Resize the per-RoI mask (typically 28x28) to the spatial extent of the
   detected bounding box.
4. Paste the resized mask into a canvas matching the input image size.
5. Threshold probabilities to obtain final binary masks.

When multiple instances overlap, pixels are assigned to the instance with the
highest detection score to prevent duplicated labels in the final output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

Boxes = Float[Array, "N 4"]
Logits = Float[Array, "N C H W"]
ClassAgnosticLogits = Float[Array, "N H W"]
Labels = Int[Array, "N"]
Scores = Float[Array, "N"]
Masks = Bool[Array, "N H W"]


def _validate_inputs(
    mask_logits: jnp.ndarray,
    boxes: jnp.ndarray,
    labels: jnp.ndarray,
    image_shape: Sequence[int],
) -> tuple[int, int, int]:
    if mask_logits.ndim not in (3, 4):
        raise ValueError("mask_logits must have shape (num_instances, num_classes, H, W) or (num_instances, H, W) for class-agnostic predictions.")

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("detected boxes must have shape (num_instances, 4).")

    if boxes.shape[0] != mask_logits.shape[0]:
        raise ValueError("Number of detected boxes must match number of mask predictions.")

    if mask_logits.ndim == 4 and labels.shape[0] != mask_logits.shape[0]:
        raise ValueError("labels must contain one entry per instance when masks are class-specific.")

    if len(image_shape) != 2:
        raise ValueError("image_shape must be a sequence of (height, width).")

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive.")

    return mask_logits.shape[0], image_height, image_width


def _select_class_masks(
    mask_logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    if mask_logits.ndim == 3:
        return mask_logits

    num_instances, _, height, width = mask_logits.shape
    labels = jnp.asarray(labels, dtype=jnp.int32)
    labels = jnp.clip(labels, 0, mask_logits.shape[1] - 1)
    indices = labels[:, None, None, None]
    indices = jnp.broadcast_to(indices, (num_instances, 1, height, width))
    selected = jnp.take_along_axis(mask_logits, indices, axis=1)
    return jnp.squeeze(selected, axis=1)


def _resize_mask(
    mask: jnp.ndarray,
    target_height: int,
    target_width: int,
) -> jnp.ndarray:
    if target_height <= 0 or target_width <= 0:
        return jnp.zeros((target_height, target_width), dtype=mask.dtype)

    # jax.image.resize expects float inputs.
    from jax import image as jimage

    return jimage.resize(
        mask,
        shape=(target_height, target_width),
        method="bilinear",
        antialias=False,
    )


def postprocess_masks(
    mask_logits: Logits | ClassAgnosticLogits,
    detected_boxes: Mapping[str, Array],
    image_shape: Sequence[int],
    threshold: float = 0.5,
) -> Masks:
    """Convert predicted mask logits into binary instance masks.

    Args:
        mask_logits: Mask predictions with shape ``(N, C, H, W)`` for
            class-specific heads or ``(N, H, W)`` for class-agnostic heads.
        detected_boxes: Mapping containing detection metadata. Expected keys:
            - ``"boxes"``: array of shape ``(N, 4)`` in ``(x1, y1, x2, y2)``.
            - ``"labels"``: array of shape ``(N,)`` with class indices. Only
              required for class-specific masks.
            - ``"scores"``: optional array of shape ``(N,)`` used to break
              ties in overlapping regions (higher score takes precedence).
        image_shape: Sequence ``(height, width)`` describing the image size.
        threshold: Probability threshold applied after sigmoid activation.

    Returns:
        Boolean array of shape ``(N, height, width)`` with one binary mask per
        detected instance.
    """
    logits = jnp.asarray(mask_logits, dtype=jnp.float32)

    boxes_value = detected_boxes.get("boxes")
    if boxes_value is None:
        raise ValueError("detected_boxes must provide a 'boxes' entry.")
    boxes = jnp.asarray(boxes_value, dtype=jnp.float32)

    labels_value = detected_boxes.get("labels")
    if logits.ndim == 4 and labels_value is None:
        raise ValueError("detected_boxes must provide 'labels' for class-specific masks.")
    if labels_value is None:
        labels_value = jnp.zeros((boxes.shape[0],), dtype=jnp.int32)
    labels = jnp.asarray(labels_value, dtype=jnp.int32)

    scores = detected_boxes.get("scores")
    scores_array = jnp.asarray(scores, dtype=jnp.float32) if scores is not None else None

    num_instances, image_height, image_width = _validate_inputs(
        logits,
        boxes,
        labels,
        image_shape,
    )

    if num_instances == 0:
        return jnp.zeros((0, image_height, image_width), dtype=jnp.bool_)

    per_class_logits = _select_class_masks(logits, labels)
    probabilities = jnn.sigmoid(per_class_logits)

    x1 = jnp.floor(boxes[:, 0]).astype(jnp.int32)
    y1 = jnp.floor(boxes[:, 1]).astype(jnp.int32)
    x2 = jnp.ceil(boxes[:, 2]).astype(jnp.int32)
    y2 = jnp.ceil(boxes[:, 3]).astype(jnp.int32)

    x1 = jnp.clip(x1, 0, image_width)
    y1 = jnp.clip(y1, 0, image_height)
    x2 = jnp.clip(x2, 0, image_width)
    y2 = jnp.clip(y2, 0, image_height)

    order = jnp.argsort(scores_array)[::-1] if scores_array is not None else jnp.arange(num_instances)

    occupied = jnp.zeros((image_height, image_width), dtype=jnp.bool_)
    masks: list[jnp.ndarray] = [jnp.zeros((image_height, image_width), dtype=jnp.bool_) for _ in range(num_instances)]

    for idx in [int(i) for i in order.tolist()]:
        top = int(y1[idx])
        left = int(x1[idx])
        bottom = int(y2[idx])
        right = int(x2[idx])

        if bottom <= top or right <= left:
            continue

        roi_height = bottom - top
        roi_width = right - left

        resized = _resize_mask(probabilities[idx], roi_height, roi_width)
        binary_roi = resized >= threshold

        occupied_region = occupied[top:bottom, left:right]
        binary_roi = jnp.logical_and(binary_roi, jnp.logical_not(occupied_region))

        current_mask = masks[idx]
        current_mask = current_mask.at[top:bottom, left:right].set(binary_roi)
        masks[idx] = current_mask

        updated_region = jnp.logical_or(occupied_region, binary_roi)
        occupied = occupied.at[top:bottom, left:right].set(updated_region)

    return jnp.stack(masks, axis=0).astype(jnp.bool_)


__all__ = ["postprocess_masks"]
