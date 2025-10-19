"""Tests for IoU computations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from detectax.models.utils import box_iou, giou


def test_perfect_overlap_returns_one() -> None:
    boxes = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    result = box_iou(boxes, boxes)
    np.testing.assert_allclose(result, np.ones((1, 1), dtype=np.float32), rtol=1e-5, atol=1e-6)


def test_non_overlapping_boxes_return_zero() -> None:
    box_a = jnp.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=jnp.float32)
    box_b = jnp.asarray([[2.0, 2.0, 3.0, 3.0]], dtype=jnp.float32)
    result = box_iou(box_a, box_b)
    np.testing.assert_array_equal(result, np.zeros((1, 1), dtype=np.float32))


def test_partial_overlap_matches_expected_value() -> None:
    boxes1 = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    boxes2 = jnp.asarray([[1.0, 1.0, 3.0, 3.0]], dtype=jnp.float32)
    result = box_iou(boxes1, boxes2)
    expected = 1.0 / 7.0  # intersection=1, union=7
    np.testing.assert_allclose(result, np.full((1, 1), expected, dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_vectorized_pairwise_result_matches_manual_loop() -> None:
    boxes1 = jnp.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 4.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    boxes2 = jnp.asarray(
        [
            [0.5, 0.5, 1.5, 1.5],
            [2.0, 2.0, 3.0, 3.5],
            [4.0, 4.0, 5.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    result = np.asarray(box_iou(boxes1, boxes2))

    def manual_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        x_left = max(xa1, xb1)
        y_top = max(ya1, yb1)
        x_right = min(xa2, xb2)
        y_bottom = min(ya2, yb2)
        inter = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    expected = np.empty_like(result)
    for i, box_a in enumerate(np.asarray(boxes1)):
        for j, box_b in enumerate(np.asarray(boxes2)):
            expected[i, j] = manual_iou(box_a, box_b)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_zero_area_boxes_produce_zero_iou() -> None:
    zero_box = jnp.asarray([[1.0, 1.0, 1.0, 2.0]], dtype=jnp.float32)
    normal_box = jnp.asarray([[0.0, 0.0, 3.0, 3.0]], dtype=jnp.float32)
    result = box_iou(zero_box, normal_box)
    np.testing.assert_array_equal(result, np.zeros((1, 1), dtype=np.float32))


def test_iou_symmetry_property() -> None:
    boxes_a = jnp.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 3.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    boxes_b = jnp.asarray(
        [
            [2.0, 2.0, 4.0, 4.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    iou_ab = np.asarray(box_iou(boxes_a, boxes_b))
    iou_ba = np.asarray(box_iou(boxes_b, boxes_a))
    np.testing.assert_allclose(iou_ab, iou_ba.T, rtol=1e-6, atol=1e-6)


def test_giou_reduces_to_iou_for_identical_boxes() -> None:
    boxes = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    g = giou(boxes, boxes)
    i = box_iou(boxes, boxes)
    np.testing.assert_allclose(g, i, rtol=1e-6, atol=1e-6)
