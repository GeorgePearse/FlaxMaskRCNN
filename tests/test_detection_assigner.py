"""Tests for second-stage detection target assignment."""

from __future__ import annotations

import numpy as np
import pytest

from detectrax.models.task_modules.assigners import assign_detection_targets
from detectrax.models.utils import encode_boxes

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_positive_and_negative_assignment() -> None:
    """Proposals with sufficient IoU receive foreground labels."""
    proposals = jnp.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 20.0, 20.0],
            [30.0, 30.0, 40.0, 40.0],
        ],
        dtype=jnp.float32,
    )
    gt_boxes = jnp.array(
        [
            [0.0, 0.0, 11.0, 11.0],
            [10.0, 10.0, 21.0, 21.0],
        ],
        dtype=jnp.float32,
    )
    gt_labels = jnp.array([1, 2], dtype=jnp.int32)

    labels, target_deltas, weights = assign_detection_targets(proposals, gt_boxes, gt_labels)

    np.testing.assert_array_equal(labels, np.array([1, 2, 0], dtype=np.int32))
    assert target_deltas.shape == (3, 3, 4)
    assert weights.shape == (3, 3, 4)
    assert jnp.all(weights[2] == 0.0)


def test_background_assignment_with_empty_gt() -> None:
    """When no ground truth is present every proposal becomes background."""
    proposals = jnp.array(
        [
            [5.0, 5.0, 15.0, 15.0],
            [25.0, 25.0, 35.0, 35.0],
        ],
        dtype=jnp.float32,
    )
    gt_boxes = jnp.zeros((0, 4), dtype=jnp.float32)
    gt_labels = jnp.zeros((0, 3), dtype=jnp.float32)

    labels, target_deltas, weights = assign_detection_targets(proposals, gt_boxes, gt_labels)

    np.testing.assert_array_equal(labels, np.array([0, 0], dtype=np.int32))
    assert target_deltas.shape == (2, 3, 4)
    assert jnp.all(target_deltas == 0.0)
    assert jnp.all(weights == 0.0)


def test_target_delta_matches_box_encoder() -> None:
    """Regression targets should match the Faster R-CNN encoding."""
    proposal = jnp.array([[0.0, 0.0, 10.0, 10.0]], dtype=jnp.float32)
    gt_box = jnp.array([[1.0, 1.0, 12.0, 12.0]], dtype=jnp.float32)
    gt_labels = jnp.array([2], dtype=jnp.int32)

    labels, target_deltas, weights = assign_detection_targets(proposal, gt_box, gt_labels)

    expected_delta = encode_boxes(gt_box, proposal)
    np.testing.assert_allclose(target_deltas[0, 2], expected_delta[0])
    assert jnp.all(weights[0, 2] == 1.0)


def test_class_specific_targets_under_vmap() -> None:
    """Vectorised assignment should retain class-specific regression targets."""
    proposals = jnp.array(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [15.0, 15.0, 25.0, 25.0],
            ],
            [
                [5.0, 5.0, 15.0, 15.0],
                [30.0, 30.0, 40.0, 40.0],
            ],
        ],
        dtype=jnp.float32,
    )
    gt_boxes = jnp.array(
        [
            [[0.0, 0.0, 10.0, 10.0]],
            [[30.0, 30.0, 42.0, 42.0]],
        ],
        dtype=jnp.float32,
    )
    gt_labels = jnp.array(
        [
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float32,
    )

    vmapped_assign = jax.vmap(assign_detection_targets, in_axes=(0, 0, 0))
    labels, target_deltas, weights = vmapped_assign(proposals, gt_boxes, gt_labels)

    np.testing.assert_array_equal(labels[0], np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(labels[1], np.array([0, 2], dtype=np.int32))

    assert jnp.all(weights[0, 0, 1] == 1.0)
    assert jnp.all(weights[0, 0, 2] == 0.0)
    assert jnp.all(weights[1, 1, 2] == 1.0)
    assert jnp.all(weights[1, 1, 1] == 0.0)
