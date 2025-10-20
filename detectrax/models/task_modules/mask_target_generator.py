"""Mask target generation utilities for Mask R-CNN training.

This module implements the logic required to convert ground-truth instance
masks into fixed-size targets aligned with positive Region of Interest (RoI)
proposals. For each positive proposal we:

1. Match it to the best-overlapping ground-truth annotation via IoU.
2. Decode the corresponding mask (supporting bitmap, polygon, and RLE inputs).
3. Crop the mask to the proposal box, padding empty regions with zeros.
4. Resize the crop to ``mask_size`` × ``mask_size`` using bilinear sampling.

The resulting targets are returned as ``float32`` JAX arrays that can be used
as supervision for the mask head during training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import image as jimage
from jaxtyping import Array, Float
from pycocotools import mask as mask_utils

from detectrax.models.utils import box_iou

# Type alias for the variety of mask annotations we accept.
MaskAnnotation = Any


def generate_mask_targets(
    positive_proposals: Float[Array, "num_pos 4"],
    gt_boxes: Float[Array, "num_gt 4"],
    gt_masks: Sequence[MaskAnnotation],
    mask_size: int = 28,
) -> Float[Array, "num_pos mask_size mask_size"]:
    """Generate fixed-size mask targets for positive proposals.

    Args:
        positive_proposals: Positive proposal boxes in ``(x1, y1, x2, y2)``
            pixel coordinates. The array shape is ``(N, 4)`` where ``N`` is the
            number of positive proposals.
        gt_boxes: Ground-truth bounding boxes for the image in the same format
            as ``positive_proposals`` with shape ``(M, 4)``.
        gt_masks: Sequence of ground-truth mask annotations (length ``M``).
            Each annotation may be provided as a binary array, COCO-style RLE,
            or polygon representation.
        mask_size: Target spatial resolution (default ``28``).

    Returns:
        A ``(N, mask_size, mask_size)`` array containing the resized mask
        targets aligned with each positive proposal. If no positive proposals
        are provided, an empty array with zero rows is returned.
    """

    if mask_size <= 0:
        raise ValueError(f"mask_size must be positive, received {mask_size}.")

    proposals_np = np.asarray(positive_proposals, dtype=np.float32)
    gt_boxes_np = np.asarray(gt_boxes, dtype=np.float32)

    if proposals_np.ndim != 2 or proposals_np.shape[1] != 4:
        raise ValueError(f"positive_proposals must have shape (N, 4); received {proposals_np.shape}.")
    if gt_boxes_np.ndim != 2 or gt_boxes_np.shape[1] != 4:
        raise ValueError(f"gt_boxes must have shape (M, 4); received {gt_boxes_np.shape}.")
    if len(gt_masks) != gt_boxes_np.shape[0]:
        raise ValueError(
            f"gt_masks must contain the same number of entries as gt_boxes; received {len(gt_masks)} masks for {gt_boxes_np.shape[0]} boxes."
        )

    num_pos = proposals_np.shape[0]
    if num_pos == 0:
        return jnp.zeros((0, mask_size, mask_size), dtype=jnp.float32)

    num_gt = gt_boxes_np.shape[0]
    if num_gt == 0:
        return jnp.zeros((num_pos, mask_size, mask_size), dtype=jnp.float32)

    matched_indices, matched_scores = _match_proposals_to_gt(proposals_np, gt_boxes_np)
    canvas_shape = _infer_canvas_shape(proposals_np, gt_boxes_np)

    cache: dict[int, np.ndarray] = {}
    targets: list[jnp.ndarray] = []

    for proposal_idx in range(num_pos):
        iou_score = matched_scores[proposal_idx]
        if iou_score <= 0.0:
            targets.append(jnp.zeros((mask_size, mask_size), dtype=jnp.float32))
            continue

        gt_index = int(matched_indices[proposal_idx])
        if gt_index not in cache:
            cache[gt_index] = _decode_mask(gt_masks[gt_index], canvas_shape)

        gt_mask_bitmap = cache[gt_index]
        cropped = _crop_mask_to_roi(gt_mask_bitmap, proposals_np[proposal_idx])
        resized = _resize_to_target(cropped, mask_size)
        targets.append(resized)

    return jnp.stack(targets, axis=0)


def _match_proposals_to_gt(
    proposals: np.ndarray,
    gt_boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best-matching GT index and IoU score for each proposal."""

    ious = np.asarray(box_iou(jnp.asarray(proposals), jnp.asarray(gt_boxes)))
    best_indices = np.argmax(ious, axis=1)
    best_scores = np.max(ious, axis=1)
    return best_indices.astype(np.int32), best_scores.astype(np.float32)


def _infer_canvas_shape(
    proposals: np.ndarray,
    gt_boxes: np.ndarray,
) -> tuple[int, int]:
    """Infer a plausible (height, width) canvas that contains all boxes."""

    if proposals.size == 0 and gt_boxes.size == 0:
        return (1, 1)

    max_x = 0.0
    max_y = 0.0
    for boxes in (proposals, gt_boxes):
        if boxes.size == 0:
            continue
        max_x = max(max_x, float(np.max(boxes[:, 2], initial=0.0)))
        max_y = max(max_y, float(np.max(boxes[:, 3], initial=0.0)))

    width = max(1, int(np.ceil(max_x)))
    height = max(1, int(np.ceil(max_y)))
    return (height, width)


def _decode_mask(mask: MaskAnnotation, fallback_shape: tuple[int, int]) -> np.ndarray:
    """Decode a mask annotation to a binary ``float32`` numpy array."""

    if isinstance(mask, jnp.ndarray):
        return np.asarray(mask, dtype=np.float32)
    if isinstance(mask, np.ndarray):
        return mask.astype(np.float32)

    if isinstance(mask, Mapping):
        if "segmentation" in mask:
            return _decode_mask(mask["segmentation"], _resolve_size(mask, fallback_shape))
        if "counts" in mask:
            rle = mask.copy()
            if "size" not in rle:
                rle["size"] = _resolve_size(mask, fallback_shape)
            decoded = mask_utils.decode(rle)
            return decoded.astype(np.float32)
        if "polygons" in mask:
            polygons = mask["polygons"]
            height, width = _resolve_size(mask, fallback_shape)
            return _decode_polygon(polygons, height, width)
        if "poly" in mask:
            polygons = mask["poly"]
            height, width = _resolve_size(mask, fallback_shape)
            return _decode_polygon(polygons, height, width)

    if isinstance(mask, list | tuple):
        if len(mask) == 0:
            height, width = fallback_shape
            return np.zeros((height, width), dtype=np.float32)

        first = mask[0]
        if isinstance(first, list | tuple):
            height, width = fallback_shape
            return _decode_polygon(mask, height, width)
        if isinstance(first, float | int):
            height, width = fallback_shape
            return _decode_polygon([mask], height, width)

    raise TypeError(f"Unsupported mask annotation type: {type(mask)!r}")


def _resolve_size(annotation: Mapping[str, Any], fallback_shape: tuple[int, int]) -> tuple[int, int]:
    """Extract (height, width) from the annotation with fallback."""

    if "size" in annotation:
        size = annotation["size"]
        if isinstance(size, list | tuple) and len(size) == 2:
            return int(size[0]), int(size[1])
    height = annotation.get("height")
    width = annotation.get("width")
    if height is not None and width is not None:
        return int(height), int(width)
    if "image_size" in annotation:
        image_size = annotation["image_size"]
        if isinstance(image_size, list | tuple) and len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])
    return fallback_shape


def _decode_polygon(polygons: Sequence[Sequence[float]], height: int, width: int) -> np.ndarray:
    """Decode polygon annotations to a binary mask using pycocotools."""

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask size inferred: {(height, width)}")
    if not polygons:
        return np.zeros((height, width), dtype=np.float32)

    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    decoded = mask_utils.decode(rle)
    return decoded.astype(np.float32)


def _crop_mask_to_roi(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop the mask to the region described by ``box`` with zero padding."""

    if mask.ndim != 2:
        raise ValueError(f"Decoded mask must be 2D, received shape {mask.shape}.")

    height, width = mask.shape
    x1, y1, x2, y2 = box.astype(np.float32)

    x1_int = int(np.floor(x1))
    y1_int = int(np.floor(y1))
    x2_int = int(np.ceil(x2))
    y2_int = int(np.ceil(y2))

    roi_width = max(1, x2_int - x1_int)
    roi_height = max(1, y2_int - y1_int)

    output = np.zeros((roi_height, roi_width), dtype=np.float32)

    clipped_x1 = np.clip(x1_int, 0, width)
    clipped_y1 = np.clip(y1_int, 0, height)
    clipped_x2 = np.clip(x2_int, 0, width)
    clipped_y2 = np.clip(y2_int, 0, height)

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return output

    dest_x1 = clipped_x1 - x1_int
    dest_y1 = clipped_y1 - y1_int
    dest_x2 = dest_x1 + (clipped_x2 - clipped_x1)
    dest_y2 = dest_y1 + (clipped_y2 - clipped_y1)

    output[dest_y1:dest_y2, dest_x1:dest_x2] = mask[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
    return output


def _resize_to_target(mask: np.ndarray, mask_size: int) -> jnp.ndarray:
    """Resize the cropped mask to ``mask_size`` × ``mask_size`` using bilinear interpolation."""

    if mask.size == 0:
        return jnp.zeros((mask_size, mask_size), dtype=jnp.float32)

    mask_jnp = jnp.asarray(mask, dtype=jnp.float32)
    if mask_jnp.shape == (mask_size, mask_size):
        return mask_jnp

    resized = jimage.resize(mask_jnp, (mask_size, mask_size), method="linear", antialias=False)
    return jnp.clip(resized, 0.0, 1.0)
