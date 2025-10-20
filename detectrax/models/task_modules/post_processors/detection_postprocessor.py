"""Detection post-processing utilities.

This module transforms class scores and box regression outputs from the
second-stage detection head into final image-level detections. The procedure
closely follows the Faster/Mask R-CNN inference recipe:

* Apply softmax to classification logits to obtain probabilities.
* Decode box regression deltas relative to the proposal boxes.
* Filter detections by a score threshold to remove low-confidence boxes.
* Run per-class non-maximum suppression (NMS) to remove redundant boxes.
* Select the top ``max_per_image`` detections per image.

The implementation supports batched inputs and returns per-image results to
avoid padding with sentinel values. Outputs are provided as JAX arrays to
facilitate downstream metric computation without leaving the JAX ecosystem.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from detectrax.models.utils.box_coder import decode_boxes
from detectrax.models.utils.nms import nms

_NUM_DETECTIONS = "num_detections"
_NUM_DETECTIONS_WITH_COORDS = "num_detections 4"

ProposalArray = Float[Array, "batch num_boxes 4"]
ClsScoreArray = Float[Array, "batch num_boxes num_classes"]
BoxDeltaArray = Float[Array, "batch num_boxes num_classes_times4"]
BoxesList = list[Float[Array, _NUM_DETECTIONS_WITH_COORDS]]
ScoresList = list[Float[Array, _NUM_DETECTIONS]]
LabelsList = list[Int[Array, _NUM_DETECTIONS]]


def _prepare_image_shapes(image_shape: Sequence[int] | Sequence[Sequence[int]] | jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Validate and broadcast image shapes to ``[batch, 2]`` (height, width)."""
    shape_array = jnp.asarray(image_shape, dtype=jnp.float32)
    if shape_array.ndim == 1:
        if shape_array.shape[0] != 2:
            raise ValueError(f"image_shape must contain (height, width); received {shape_array.shape}.")
        shape_array = jnp.broadcast_to(shape_array[None, :], (batch_size, 2))
    elif shape_array.ndim == 2:
        if shape_array.shape != (batch_size, 2):
            raise ValueError(f"image_shape with ndim=2 must have shape (batch, 2); received {shape_array.shape} for batch size {batch_size}.")
    else:
        raise ValueError(f"image_shape must be a 1D/2D array or sequence; received ndim={shape_array.ndim}.")
    return shape_array


def _clip_boxes_to_image(boxes: jnp.ndarray, height: float, width: float) -> jnp.ndarray:
    """Clip ``(x1, y1, x2, y2)`` boxes to the image boundaries."""
    x1 = jnp.clip(boxes[..., 0], min=0.0, max=width)
    y1 = jnp.clip(boxes[..., 1], min=0.0, max=height)
    x2 = jnp.clip(boxes[..., 2], min=0.0, max=width)
    y2 = jnp.clip(boxes[..., 3], min=0.0, max=height)
    x1 = jnp.minimum(x1, x2)
    y1 = jnp.minimum(y1, y2)
    return jnp.stack((x1, y1, x2, y2), axis=-1)


def _gather_valid_indices(indices: jnp.ndarray, valid_count: jnp.ndarray) -> jnp.ndarray:
    """Extract the valid NMS indices (strip ``-1`` padding)."""
    if indices.ndim != 1:
        raise ValueError(f"NMS indices must be 1D; received shape {indices.shape}.")
    valid = int(jnp.asarray(valid_count))
    if valid == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    return jnp.asarray(indices[:valid], dtype=jnp.int32)


def postprocess_detections(
    proposals: ProposalArray,
    cls_scores: ClsScoreArray,
    box_deltas: BoxDeltaArray,
    image_shape: Sequence[int] | Sequence[Sequence[int]] | jnp.ndarray,
    *,
    score_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    max_per_image: int = 100,
) -> tuple[BoxesList, ScoresList, LabelsList]:
    """Convert detection head outputs into final per-image detections.

    Args:
        proposals: Proposal boxes for each image shaped ``[batch, num_boxes, 4]``.
        cls_scores: Classification logits shaped ``[batch, num_boxes, num_classes]``.
        box_deltas: Box regression deltas shaped ``[batch, num_boxes, num_classes * 4]``.
        image_shape: Image height/width provided as ``(H, W)`` or ``[(H_i, W_i), ...]``.
        score_threshold: Minimum class probability required to keep a detection.
        nms_threshold: IoU threshold for per-class non-maximum suppression.
        max_per_image: Maximum number of detections to return per image.

    Returns:
        Tuple ``(boxes, scores, labels)`` where each element is a list of length
        ``batch`` containing variable-length JAX arrays for that image.

    Raises:
        ValueError: If inputs have incompatible shapes or unsupported ranks.
    """
    proposals = jnp.asarray(proposals, dtype=jnp.float32)
    cls_scores = jnp.asarray(cls_scores, dtype=jnp.float32)
    box_deltas = jnp.asarray(box_deltas, dtype=jnp.float32)

    if proposals.ndim != 3 or proposals.shape[-1] != 4:
        raise ValueError(f"proposals must have shape (batch, num_boxes, 4); received {proposals.shape}.")
    if cls_scores.shape[:2] != proposals.shape[:2]:
        raise ValueError(f"cls_scores must align with proposals on batch/box dimensions; received {cls_scores.shape} vs {proposals.shape}.")
    if cls_scores.ndim != 3:
        raise ValueError(f"cls_scores must have shape (batch, num_boxes, num_classes); received {cls_scores.shape}.")
    if box_deltas.ndim != 3 or box_deltas.shape[:2] != proposals.shape[:2]:
        raise ValueError(f"box_deltas must have shape (batch, num_boxes, num_classes * 4); received {box_deltas.shape}.")

    batch_size, num_boxes, _ = proposals.shape
    num_classes = cls_scores.shape[-1]
    if box_deltas.shape[-1] != num_classes * 4:
        raise ValueError(f"Expected box_deltas last dimension to be num_classes * 4 ({num_classes * 4}); received {box_deltas.shape[-1]}.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2 (including background class).")

    image_shapes = _prepare_image_shapes(image_shape, batch_size)
    class_probabilities = jax.nn.softmax(cls_scores, axis=-1)

    deltas = box_deltas.reshape(batch_size, num_boxes, num_classes, 4)
    anchors = jnp.broadcast_to(proposals[:, :, None, :], deltas.shape)
    decoded_boxes = decode_boxes(deltas.reshape(-1, 4), anchors.reshape(-1, 4)).reshape(deltas.shape)

    boxes_per_image: BoxesList = []
    scores_per_image: ScoresList = []
    labels_per_image: LabelsList = []

    for image_idx in range(batch_size):
        height, width = image_shapes[image_idx]
        per_class_boxes = _clip_boxes_to_image(decoded_boxes[image_idx], float(height), float(width))
        per_class_scores = class_probabilities[image_idx]

        kept_boxes = []
        kept_scores = []
        kept_labels = []

        for class_id in range(1, num_classes):
            scores_for_class = per_class_scores[:, class_id]
            score_mask = scores_for_class > score_threshold
            if not bool(jnp.any(score_mask)):
                continue

            candidate_indices = jnp.nonzero(score_mask, size=None)[0]
            candidate_boxes = jnp.take(per_class_boxes[:, class_id, :], candidate_indices, axis=0)
            candidate_scores = jnp.take(scores_for_class, candidate_indices, axis=0)

            if candidate_boxes.shape[0] == 0:
                continue

            nms_result = nms(
                candidate_boxes,
                candidate_scores,
                iou_threshold=nms_threshold,
                max_output_size=max_per_image,
            )
            valid_indices = _gather_valid_indices(nms_result.indices, nms_result.valid_counts)
            if valid_indices.shape[0] == 0:
                continue

            selected_boxes = jnp.take(candidate_boxes, valid_indices, axis=0)
            selected_scores = jnp.take(candidate_scores, valid_indices, axis=0)
            selected_labels = jnp.full((valid_indices.shape[0],), class_id, dtype=jnp.int32)

            kept_boxes.append(selected_boxes)
            kept_scores.append(selected_scores)
            kept_labels.append(selected_labels)

        if not kept_boxes:
            boxes_per_image.append(jnp.zeros((0, 4), dtype=jnp.float32))
            scores_per_image.append(jnp.zeros((0,), dtype=jnp.float32))
            labels_per_image.append(jnp.zeros((0,), dtype=jnp.int32))
            continue

        all_boxes = jnp.concatenate(kept_boxes, axis=0)
        all_scores = jnp.concatenate(kept_scores, axis=0)
        all_labels = jnp.concatenate(kept_labels, axis=0)

        total_detections = all_scores.shape[0]
        if total_detections > max_per_image:
            top_k = min(max_per_image, total_detections)
            top_scores, top_indices = jax.lax.top_k(all_scores, top_k)
            all_boxes = jnp.take(all_boxes, top_indices, axis=0)
            all_labels = jnp.take(all_labels, top_indices, axis=0)
            all_scores = top_scores

        boxes_per_image.append(all_boxes)
        scores_per_image.append(all_scores)
        labels_per_image.append(all_labels)

    return boxes_per_image, scores_per_image, labels_per_image


__all__ = ["postprocess_detections"]
