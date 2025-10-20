"""Tests for the Region Proposal Network loss."""

from __future__ import annotations

import numpy as np
import pytest

from detectax.models.losses import rpn_loss

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")


def test_objectness_binary_cross_entropy_matches_expected() -> None:
    """Classification loss should match the BCE computed over sampled anchors."""
    objectness_pred = jnp.array([[0.0, 1.5, -1.5, 0.2]], dtype=jnp.float32)
    objectness_targets = jnp.array([[1, 0, 1, 0]], dtype=jnp.int32)
    box_deltas_pred = jnp.zeros((1, 4, 4), dtype=jnp.float32)
    box_delta_targets = jnp.zeros_like(box_deltas_pred)
    weights = jnp.ones((1, 4), dtype=jnp.float32)

    total_loss, cls_loss, reg_loss = rpn_loss(objectness_pred, box_deltas_pred, objectness_targets, box_delta_targets, weights)

    labels = objectness_targets.astype(jnp.float32)
    expected_cls = jnp.mean(optax.sigmoid_binary_cross_entropy(objectness_pred, labels))

    np.testing.assert_allclose(reg_loss, 0.0, atol=1e-7)
    np.testing.assert_allclose(cls_loss, expected_cls, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(total_loss, cls_loss, rtol=1e-6, atol=1e-6)


def test_box_regression_loss_only_on_positive_anchors() -> None:
    """Regression loss should accumulate Smooth L1 errors for positives only."""
    objectness_pred = jnp.zeros((1, 4), dtype=jnp.float32)
    objectness_targets = jnp.array([[1, 0, 1, 0]], dtype=jnp.int32)
    box_deltas_pred = jnp.array(
        [
            [
                [0.2, -0.1, 0.5, -0.3],
                [0.0, 0.0, 0.0, 0.0],
                [1.5, 0.0, -2.0, 0.5],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    box_delta_targets = jnp.array(
        [
            [
                [0.0, 0.1, 0.3, -0.1],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -0.5, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    weights = jnp.ones((1, 4), dtype=jnp.float32)

    total_loss, cls_loss, reg_loss = rpn_loss(objectness_pred, box_deltas_pred, objectness_targets, box_delta_targets, weights)

    labels = objectness_targets.astype(jnp.float32)
    expected_cls = jnp.mean(optax.sigmoid_binary_cross_entropy(objectness_pred, labels))

    diff = jnp.abs(box_deltas_pred - box_delta_targets)
    smooth_l1 = jnp.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)
    reg_per_anchor = jnp.sum(smooth_l1, axis=-1)
    positive_mask = objectness_targets.astype(bool)
    expected_reg = jnp.mean(reg_per_anchor[positive_mask])

    np.testing.assert_allclose(cls_loss, expected_cls, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(reg_loss, expected_reg, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(total_loss, expected_cls + expected_reg, rtol=1e-6, atol=1e-6)


def test_sample_balancing_enforces_pos_neg_ratio() -> None:
    """Exactly 256 anchors (128 positive/negative) should contribute when available."""
    num_anchors = 400
    objectness_pred = jnp.zeros((1, num_anchors), dtype=jnp.float32)
    objectness_targets = jnp.concatenate(
        (jnp.ones((1, num_anchors // 2), dtype=jnp.int32), jnp.zeros((1, num_anchors // 2), dtype=jnp.int32)),
        axis=1,
    )
    box_deltas_pred = jnp.zeros((1, num_anchors, 4), dtype=jnp.float32)
    box_delta_targets = jnp.zeros_like(box_deltas_pred)
    weights = jnp.ones((1, num_anchors), dtype=jnp.float32)

    def loss_fn(logits: jnp.ndarray) -> jnp.ndarray:
        total, _, _ = rpn_loss(logits, box_deltas_pred, objectness_targets, box_delta_targets, weights)
        return total

    grads = jax.grad(loss_fn)(objectness_pred)
    grad_values = np.asarray(grads[0])
    pos_grads = grad_values[: num_anchors // 2]
    neg_grads = grad_values[num_anchors // 2 :]

    nonzero_pos = pos_grads[np.abs(pos_grads) > 1e-8]
    nonzero_neg = neg_grads[np.abs(neg_grads) > 1e-8]

    assert nonzero_pos.shape[0] == 128
    assert nonzero_neg.shape[0] == 128

    expected_positive_grad = -0.5 / 256.0
    expected_negative_grad = 0.5 / 256.0

    np.testing.assert_allclose(nonzero_pos, expected_positive_grad, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(nonzero_neg, expected_negative_grad, rtol=1e-6, atol=1e-9)


def test_gradients_flow_through_regression_branch() -> None:
    """Positive anchors should receive regression gradients while negatives stay zero."""
    objectness_pred = jnp.zeros((1, 2), dtype=jnp.float32)
    objectness_targets = jnp.array([[1, 0]], dtype=jnp.int32)
    box_deltas_pred = jnp.array(
        [[[0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    box_delta_targets = jnp.zeros_like(box_deltas_pred)
    weights = jnp.ones((1, 2), dtype=jnp.float32)

    def loss_fn(deltas: jnp.ndarray) -> jnp.ndarray:
        total, _, _ = rpn_loss(objectness_pred, deltas, objectness_targets, box_delta_targets, weights)
        return total

    grads = jax.grad(loss_fn)(box_deltas_pred)
    positive_grad = grads[0, 0]
    negative_grad = grads[0, 1]

    assert jnp.any(jnp.abs(positive_grad) > 1e-6), "Positive anchor should receive regression gradients."
    np.testing.assert_allclose(negative_grad, 0.0, atol=1e-8)
