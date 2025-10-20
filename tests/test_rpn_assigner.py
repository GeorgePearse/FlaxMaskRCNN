"""Tests for RPN target assignment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from detectax.models.task_modules.assigners import assign_rpn_targets
from detectax.models.utils.box_coder import encode_boxes


def test_positive_assignment_for_high_iou() -> None:
    anchors = jnp.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [2.5, 2.5, 3.5, 3.5],
        ],
        dtype=jnp.float32,
    )
    gt_boxes = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    gt_labels = jnp.asarray([1], dtype=jnp.int32)

    labels, deltas, weights = assign_rpn_targets(anchors, gt_boxes, gt_labels)

    np.testing.assert_array_equal(np.asarray(labels), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(weights), np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(deltas)[1], np.zeros(4, dtype=np.float32))


def test_negative_assignment_when_iou_below_threshold() -> None:
    anchors = jnp.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=jnp.float32)
    gt_boxes = jnp.asarray([[0.6, 0.6, 1.6, 1.6]], dtype=jnp.float32)
    gt_labels = jnp.asarray([1], dtype=jnp.int32)

    labels, deltas, weights = assign_rpn_targets(anchors, gt_boxes, gt_labels)

    np.testing.assert_array_equal(np.asarray(labels), np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(weights), np.array([1.0], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(deltas), np.zeros((1, 4), dtype=np.float32))


def test_ignore_assignment_for_intermediate_iou() -> None:
    anchors = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    gt_boxes = jnp.asarray([[0.5, 0.5, 2.0, 2.0]], dtype=jnp.float32)
    gt_labels = jnp.asarray([1], dtype=jnp.int32)

    labels, _, weights = assign_rpn_targets(anchors, gt_boxes, gt_labels)

    np.testing.assert_array_equal(np.asarray(labels), np.array([-1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(weights), np.array([0.0], dtype=np.float32))


def test_empty_gt_marks_all_negative() -> None:
    anchors = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    gt_boxes = jnp.empty((0, 4), dtype=jnp.float32)
    gt_labels = jnp.empty((0,), dtype=jnp.int32)

    labels, deltas, weights = assign_rpn_targets(anchors, gt_boxes, gt_labels)

    np.testing.assert_array_equal(np.asarray(labels), np.zeros(2, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(weights), np.ones(2, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(deltas), np.zeros((2, 4), dtype=np.float32))


def test_crowd_regions_are_ignored() -> None:
    anchors = jnp.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        dtype=jnp.float32,
    )
    gt_boxes = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    gt_labels = jnp.asarray([-1], dtype=jnp.int32)

    labels, _, weights = assign_rpn_targets(anchors, gt_boxes, gt_labels)

    np.testing.assert_array_equal(np.asarray(labels), np.array([-1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(weights), np.array([0.0, 1.0], dtype=np.float32))


def test_batched_assignment_with_vmap() -> None:
    anchors = jnp.asarray(
        [
            [
                [0.0, 0.0, 2.0, 2.0],
                [2.5, 2.5, 3.5, 3.5],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
            ],
        ],
        dtype=jnp.float32,
    )
    gt_boxes = jnp.asarray(
        [
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
            ],
        ],
        dtype=jnp.float32,
    )
    gt_labels = jnp.asarray(
        [
            [1, -1],
            [-1, 2],
        ],
        dtype=jnp.int32,
    )

    batched_assign = jax.vmap(assign_rpn_targets, in_axes=(0, 0, 0))
    labels, deltas, weights = batched_assign(anchors, gt_boxes, gt_labels)

    np.testing.assert_array_equal(labels.shape, (2, 2))
    np.testing.assert_array_equal(deltas.shape, (2, 2, 4))
    np.testing.assert_array_equal(weights.shape, (2, 2))
    np.testing.assert_array_equal(np.asarray(labels[0]), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(labels[1]), np.array([-1, 1], dtype=np.int32))


def test_target_deltas_match_encode_boxes() -> None:
    anchors = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    gt_boxes = jnp.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.25, 0.25, 1.75, 1.75],
        ],
        dtype=jnp.float32,
    )
    gt_labels = jnp.asarray([1, 2], dtype=jnp.int32)

    labels, deltas, weights = assign_rpn_targets(anchors, gt_boxes, gt_labels)
    expected = encode_boxes(jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32), anchors)

    np.testing.assert_array_equal(np.asarray(labels), np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(weights), np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(deltas), np.asarray(expected), rtol=1e-6, atol=1e-6)
