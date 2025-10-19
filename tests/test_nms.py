"""Tests for Non-Maximum Suppression (NMS)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from detectax.models.utils.nms import nms


def _extract_kept(indices: jnp.ndarray, count: jnp.ndarray) -> np.ndarray:
    valid = int(np.asarray(count))
    return np.asarray(indices)[:valid]


def test_overlapping_boxes_are_suppressed() -> None:
    boxes = jnp.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.1, 0.1, 2.0, 2.0],
            [3.0, 3.0, 5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    scores = jnp.asarray([0.95, 0.9, 0.4], dtype=jnp.float32)

    result = nms(boxes, scores, iou_threshold=0.5)
    kept = _extract_kept(result.indices, result.valid_counts)

    np.testing.assert_array_equal(kept, np.array([0, 2], dtype=np.int32))


def test_non_overlapping_boxes_all_kept() -> None:
    boxes = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    scores = jnp.asarray([0.3, 0.6, 0.8], dtype=jnp.float32)

    result = nms(boxes, scores, iou_threshold=0.5)
    kept = _extract_kept(result.indices, result.valid_counts)

    np.testing.assert_array_equal(kept, np.array([2, 1, 0], dtype=np.int32))


def test_score_ordering_respected() -> None:
    boxes = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    scores = jnp.asarray([0.5, 0.8, 0.9], dtype=jnp.float32)

    result = nms(boxes, scores, iou_threshold=0.3, max_output_size=2)
    kept = _extract_kept(result.indices, result.valid_counts)

    np.testing.assert_array_equal(kept, np.array([2, 1], dtype=np.int32))


def test_batch_processing_handles_multiple_images() -> None:
    boxes = jnp.asarray(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.2, 1.2],
                [2.0, 2.0, 3.0, 3.0],
            ],
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.05, 1.05, 2.05, 2.05],
                [3.0, 3.0, 4.0, 4.0],
            ],
        ],
        dtype=jnp.float32,
    )
    scores = jnp.asarray(
        [
            [0.9, 0.8, 0.1],
            [0.95, 0.6, 0.7],
        ],
        dtype=jnp.float32,
    )

    result = nms(boxes, scores, iou_threshold=0.5)
    kept0 = _extract_kept(result.indices[0], result.valid_counts[0])
    kept1 = _extract_kept(result.indices[1], result.valid_counts[1])

    np.testing.assert_array_equal(kept0, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(kept1, np.array([0, 2], dtype=np.int32))


def test_empty_input_returns_no_indices() -> None:
    boxes = jnp.zeros((0, 4), dtype=jnp.float32)
    scores = jnp.zeros((0,), dtype=jnp.float32)

    result = nms(boxes, scores)
    kept = _extract_kept(result.indices, result.valid_counts)

    np.testing.assert_equal(kept.size, 0)


def test_single_box_always_kept() -> None:
    boxes = jnp.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=jnp.float32)
    scores = jnp.asarray([0.4], dtype=jnp.float32)

    result = nms(boxes, scores)
    kept = _extract_kept(result.indices, result.valid_counts)

    np.testing.assert_array_equal(kept, np.array([0], dtype=np.int32))


def test_max_output_size_limits_results() -> None:
    boxes = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 2.0, 3.0, 3.0],
            [4.0, 4.0, 5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    scores = jnp.asarray([0.9, 0.8, 0.7], dtype=jnp.float32)

    result = nms(boxes, scores, max_output_size=2)
    kept = _extract_kept(result.indices, result.valid_counts)

    np.testing.assert_array_equal(kept, np.array([0, 1], dtype=np.int32))


def test_all_boxes_overlap_with_top_box() -> None:
    boxes = jnp.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.1, 0.1, 1.9, 1.9],
            [0.2, 0.2, 1.8, 1.8],
        ],
        dtype=jnp.float32,
    )
    scores = jnp.asarray([0.99, 0.8, 0.7], dtype=jnp.float32)

    result = nms(boxes, scores, iou_threshold=0.3)
    kept = _extract_kept(result.indices, result.valid_counts)

    np.testing.assert_array_equal(kept, np.array([0], dtype=np.int32))
