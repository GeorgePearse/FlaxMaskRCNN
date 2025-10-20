"""Region Proposal Network (RPN) proposal generation utilities."""

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from detectrax.models.utils.box_coder import decode_boxes
from detectrax.models.utils.nms import nms

_NUM_ANCHORS = "num_anchors"
_NUM_ANCHORS_WITH_COORDS = "num_anchors 4"
_NUM_PROPOSALS = "num_proposals"
_NUM_PROPOSALS_WITH_COORDS = "num_proposals 4"

BatchedAnchors = Float[Array, f"batch {_NUM_ANCHORS} 4"]
BatchedScores = Float[Array, f"batch {_NUM_ANCHORS}"]
BatchedDeltas = Float[Array, f"batch {_NUM_ANCHORS} 4"]
BatchedImageShapes = Float[Array, "batch 2"]
BatchedProposals = Float[Array, f"batch {_NUM_PROPOSALS} 4"]
BatchedProposalScores = Float[Array, f"batch {_NUM_PROPOSALS}"]

_NEGATIVE_ONE: Final[int] = -1


def generate_proposals(
    anchors: BatchedAnchors,
    objectness: BatchedScores,
    deltas: BatchedDeltas,
    image_shape: BatchedImageShapes | Float[Array, 2],
    *,
    pre_nms_top_n: int = 2000,
    post_nms_top_n: int = 1000,
    nms_threshold: float = 0.7,
    min_size: float = 0.0,
) -> tuple[BatchedProposals, BatchedProposalScores]:
    """Generate region proposals from RPN outputs.

    Args:
        anchors: Anchor boxes shaped ``(batch, num_anchors, 4)``.
        objectness: Objectness logits or probabilities for each anchor
            shaped ``(batch, num_anchors)``.
        deltas: Predicted box regression deltas shaped
            ``(batch, num_anchors, 4)``.
        image_shape: Image height/width for each batch element. Accepts either
            ``(batch, 2)`` or a single ``(2,)`` vector broadcast across the
            batch as ``(height, width)``.
        pre_nms_top_n: Number of top-scoring proposals to consider before NMS.
        post_nms_top_n: Number of proposals to return after NMS.
        nms_threshold: IoU threshold used in NMS.
        min_size: Minimum proposal side length. Boxes with area smaller than
            ``min_size ** 2`` are discarded.

    Returns:
        Tuple ``(proposals, scores)`` where proposals are shaped
        ``(batch, post_nms_top_n, 4)`` and scores
        ``(batch, post_nms_top_n)``. Outputs are padded with zeros when fewer
        than ``post_nms_top_n`` proposals remain.
    """

    anchors_array = jnp.asarray(anchors, dtype=jnp.float32)
    objectness_array = jnp.asarray(objectness, dtype=jnp.float32)
    deltas_array = jnp.asarray(deltas, dtype=jnp.float32)
    image_shape_array = jnp.asarray(image_shape, dtype=jnp.float32)

    if anchors_array.ndim != 3 or anchors_array.shape[-1] != 4:
        raise ValueError(f"anchors must have shape (batch, num_anchors, 4); received {anchors_array.shape}.")
    if objectness_array.shape != anchors_array.shape[:2]:
        raise ValueError(f"objectness must have shape (batch, num_anchors) matching anchors; received {objectness_array.shape}.")
    if deltas_array.shape != anchors_array.shape:
        raise ValueError(f"deltas must have shape (batch, num_anchors, 4); received {deltas_array.shape}.")

    if image_shape_array.ndim == 1:
        if image_shape_array.shape[0] != 2:
            raise ValueError(f"image_shape of rank 1 must have length 2; received {image_shape_array.shape}.")
        image_shape_array = jnp.broadcast_to(image_shape_array[None, :], (anchors_array.shape[0], 2))
    elif image_shape_array.ndim == 2:
        if image_shape_array.shape != (anchors_array.shape[0], 2):
            raise ValueError(f"image_shape must have shape (batch, 2); received {image_shape_array.shape} for batch size {anchors_array.shape[0]}.")
    else:
        raise ValueError(f"image_shape must have rank 1 or 2 corresponding to (2,) or (batch, 2); received rank {image_shape_array.ndim}.")

    if pre_nms_top_n < 0:
        raise ValueError(f"pre_nms_top_n must be non-negative; received {pre_nms_top_n}.")
    if post_nms_top_n < 0:
        raise ValueError(f"post_nms_top_n must be non-negative; received {post_nms_top_n}.")
    if post_nms_top_n == 0:
        batch_size = anchors_array.shape[0]
        zero_boxes = jnp.zeros((batch_size, 0, 4), dtype=jnp.float32)
        zero_scores = jnp.zeros((batch_size, 0), dtype=jnp.float32)
        return zero_boxes, zero_scores

    num_anchors = anchors_array.shape[1]
    topk_pre = min(pre_nms_top_n, num_anchors)

    def _generate_single(
        single_anchors: Float[Array, _NUM_ANCHORS_WITH_COORDS],
        single_scores: Float[Array, _NUM_ANCHORS],
        single_deltas: Float[Array, _NUM_ANCHORS_WITH_COORDS],
        single_shape: Float[Array, 2],
    ) -> tuple[Float[Array, _NUM_PROPOSALS_WITH_COORDS], Float[Array, _NUM_PROPOSALS]]:
        """Generate proposals for a single image."""
        proposals = decode_boxes(single_deltas, single_anchors)

        height, width = single_shape
        clip_max = jnp.array([width, height, width, height], dtype=jnp.float32)
        proposals = jnp.clip(proposals, min=0.0, max=clip_max)

        widths = jnp.maximum(0.0, proposals[:, 2] - proposals[:, 0])
        heights = jnp.maximum(0.0, proposals[:, 3] - proposals[:, 1])
        areas = widths * heights

        valid_mask = jnp.logical_and(widths > 0.0, heights > 0.0)
        if min_size > 0.0:
            valid_mask = jnp.logical_and(valid_mask, areas >= (min_size**2))

        filtered_scores = jnp.where(valid_mask, single_scores, -jnp.inf)

        if topk_pre == 0 or proposals.shape[0] == 0:
            zero_boxes_single = jnp.zeros((post_nms_top_n, 4), dtype=jnp.float32)
            zero_scores_single = jnp.zeros((post_nms_top_n,), dtype=jnp.float32)
            return zero_boxes_single, zero_scores_single

        top_scores, top_indices = jax.lax.top_k(filtered_scores, topk_pre)
        top_boxes = proposals[top_indices]

        max_output = min(post_nms_top_n, topk_pre)
        nms_result = nms(top_boxes, top_scores, iou_threshold=nms_threshold, max_output_size=max_output)
        kept_indices = nms_result.indices
        valid_count = nms_result.valid_counts

        current_length = kept_indices.shape[0]
        if current_length < post_nms_top_n:
            pad_size = post_nms_top_n - current_length
            kept_indices = jnp.concatenate(
                [kept_indices, jnp.full((pad_size,), _NEGATIVE_ONE, dtype=jnp.int32)],
                axis=0,
            )

        clipped_indices = jnp.clip(kept_indices, 0, top_boxes.shape[0] - 1)
        gathered_boxes = top_boxes[clipped_indices]
        gathered_scores = top_scores[clipped_indices]

        base_mask = jnp.arange(post_nms_top_n, dtype=jnp.int32) < valid_count
        finite_mask = jnp.isfinite(gathered_scores)
        final_mask = jnp.logical_and(base_mask, finite_mask)

        padded_boxes = jnp.where(final_mask[:, None], gathered_boxes, 0.0)
        padded_scores = jnp.where(final_mask, gathered_scores, 0.0)

        return padded_boxes, padded_scores

    proposals_batch, scores_batch = jax.vmap(_generate_single)(anchors_array, objectness_array, deltas_array, image_shape_array)
    return proposals_batch, scores_batch


__all__ = ["generate_proposals"]
