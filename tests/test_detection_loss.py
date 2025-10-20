"""Unit tests for the Fast R-CNN detection loss."""

from __future__ import annotations

import numpy as np
import pytest

from detectax.models.losses import detection_loss

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_multiclass_cross_entropy_loss() -> None:
    """Cross-entropy should match the analytic expectation."""
    cls_scores = jnp.array(
        [
            [
                [0.5, 1.0, -0.5],
                [2.0, -1.0, -0.5],
                [0.0, 0.0, 0.0],
                [-1.0, 0.5, 1.5],
            ]
        ],
        dtype=jnp.float32,
    )
    cls_targets = jnp.array([[1, 0, 0, 2]], dtype=jnp.int32)
    weights = jnp.ones_like(cls_targets, dtype=jnp.float32)

    num_classes = cls_scores.shape[-1] - 1
    box_deltas = jnp.zeros((1, 4, num_classes * 4), dtype=jnp.float32)
    box_delta_targets = jnp.zeros((1, 4, 4), dtype=jnp.float32)

    total_loss, cls_loss, reg_loss = detection_loss(cls_scores, box_deltas, cls_targets, box_delta_targets, weights)

    log_probs = jax.nn.log_softmax(cls_scores, axis=-1)
    expected = -log_probs[0, jnp.arange(cls_targets.shape[1]), cls_targets[0]]
    expected = jnp.sum(expected * weights[0]) / jnp.sum(weights[0])

    np.testing.assert_allclose(cls_loss, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(reg_loss, 0.0, atol=1e-8)
    np.testing.assert_allclose(total_loss, cls_loss, atol=1e-6)


def test_class_specific_regression() -> None:
    """Regression loss should select the deltas that correspond to each class."""
    cls_scores = jnp.array(
        [
            [
                [-10.0, 15.0, -10.0],
                [-10.0, -10.0, 15.0],
            ]
        ],
        dtype=jnp.float32,
    )
    cls_targets = jnp.array([[1, 2]], dtype=jnp.int32)
    weights = jnp.ones_like(cls_targets, dtype=jnp.float32)

    box_deltas = jnp.array(
        [
            [
                [0.5, 0.5, 0.5, 0.5, -2.0, -2.0, -2.0, -2.0],
                [-3.0, -3.0, -3.0, -3.0, 1.5, 1.5, 1.5, 1.5],
            ]
        ],
        dtype=jnp.float32,
    )
    box_delta_targets = jnp.array(
        [
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [100.0, 100.0, 100.0, 100.0],
                ],
                [
                    [-100.0, -100.0, -100.0, -100.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        ],
        dtype=jnp.float32,
    )

    total_loss, cls_loss, reg_loss = detection_loss(cls_scores, box_deltas, cls_targets, box_delta_targets, weights)

    expected_reg = 0.5  # two positives, each contributes 0.5
    np.testing.assert_allclose(reg_loss, expected_reg, atol=1e-6)
    assert cls_loss < 1e-6
    np.testing.assert_allclose(total_loss, cls_loss + reg_loss, atol=1e-6)


def test_sample_balancing_limits_positive_fraction() -> None:
    """Sampling should cap positives at 25% of 512 and fill the rest with negatives."""
    num_classes = 2
    num_pos = 200
    num_neg = 600
    num_rois = num_pos + num_neg

    cls_scores = np.full((1, num_rois, num_classes + 1), -20.0, dtype=np.float32)
    cls_targets = np.zeros((1, num_rois), dtype=np.int32)
    weights = np.ones((1, num_rois), dtype=np.float32)
    box_deltas = np.zeros((1, num_rois, num_classes * 4), dtype=np.float32)
    box_delta_targets = np.zeros((1, num_rois, 4), dtype=np.float32)

    cls_targets[0, :num_pos] = 1

    # Well-classified positives (first 128)
    cls_scores[0, :128, 1] = 20.0
    cls_scores[0, :128, 0] = -20.0
    cls_scores[0, :128, 2] = -20.0
    box_deltas[0, :128, 0:4] = 0.5

    # Remaining positives should be ignored by the sampler
    cls_scores[0, 128:num_pos, 0] = 20.0
    cls_scores[0, 128:num_pos, 1] = -20.0
    cls_scores[0, 128:num_pos, 2] = -20.0
    box_deltas[0, 128:num_pos, 0:4] = 3.0

    # Well-classified negatives (384 of them)
    neg_start = num_pos
    neg_selected = 384
    cls_scores[0, neg_start : neg_start + neg_selected, 0] = 20.0

    # Remaining negatives should be ignored by the sampler
    cls_scores[0, neg_start + neg_selected :, 1] = 20.0

    total_loss, cls_loss, reg_loss = detection_loss(
        jnp.asarray(cls_scores),
        jnp.asarray(box_deltas),
        jnp.asarray(cls_targets),
        jnp.asarray(box_delta_targets),
        jnp.asarray(weights),
    )

    assert cls_loss < 1e-4, "High-loss samples should have been dropped by the sampler."
    np.testing.assert_allclose(reg_loss, 0.5, atol=1e-3)
    np.testing.assert_allclose(total_loss, cls_loss + reg_loss, atol=1e-4)


def test_background_samples_do_not_contribute_to_regression() -> None:
    """Background-only batches should yield zero regression loss."""
    cls_scores = jnp.zeros((1, 6, 3), dtype=jnp.float32)
    cls_targets = jnp.zeros((1, 6), dtype=jnp.int32)
    weights = jnp.ones_like(cls_targets, dtype=jnp.float32)

    box_deltas = jnp.ones((1, 6, 8), dtype=jnp.float32)
    box_delta_targets = jnp.full((1, 6, 8), 3.0, dtype=jnp.float32)

    total_loss, cls_loss, reg_loss = detection_loss(cls_scores, box_deltas, cls_targets, box_delta_targets, weights)

    expected_cls = -jax.nn.log_softmax(cls_scores, axis=-1)[0, :, 0]
    expected_cls = jnp.mean(expected_cls)

    np.testing.assert_allclose(cls_loss, expected_cls, atol=1e-6)
    np.testing.assert_allclose(reg_loss, 0.0, atol=1e-8)
    np.testing.assert_allclose(total_loss, cls_loss, atol=1e-6)
