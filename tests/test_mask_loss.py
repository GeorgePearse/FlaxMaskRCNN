"""Tests for the mask loss implementation."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from detectax.models.losses import mask_loss


def _targets(masks: jnp.ndarray, classes: jnp.ndarray) -> dict:
    return {"masks": masks, "classes": classes}


def test_per_pixel_bce_matches_log2() -> None:
    mask_pred = jnp.zeros((1, 1, 2, 1, 1), dtype=jnp.float32)
    mask_targets = _targets(
        masks=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
        classes=jnp.array([[1]], dtype=jnp.int32),
    )

    pos_mask = jnp.array([[True]])

    loss = mask_loss(mask_pred, mask_targets, pos_mask)

    np.testing.assert_allclose(loss, math.log(2.0), rtol=1e-5)


def test_only_positive_rois_contribute() -> None:
    mask_pred = jnp.array(
        [
            [
                [
                    [[0.0]],
                    [[0.0]],
                ],
                [
                    [[10.0]],
                    [[-10.0]],
                ],
            ]
        ],
        dtype=jnp.float32,
    )
    mask_targets = _targets(
        masks=jnp.ones((1, 2, 1, 1), dtype=jnp.float32),
        classes=jnp.array([[1, 1]], dtype=jnp.int32),
    )

    loss = mask_loss(mask_pred, mask_targets, jnp.array([0], dtype=jnp.int32))

    np.testing.assert_allclose(loss, math.log(2.0), rtol=1e-5)


def test_class_specific_masks_are_used() -> None:
    mask_pred = jnp.array(
        [
            [
                [
                    [[0.0]],
                    [[5.0]],
                ],
                [
                    [[1.0]],
                    [[-2.0]],
                ],
            ]
        ],
        dtype=jnp.float32,
    )
    mask_targets = _targets(
        masks=jnp.array([[[[1.0]], [[0.0]]]], dtype=jnp.float32),
        classes=jnp.array([[0, 1]], dtype=jnp.int32),
    )
    positive_mask = jnp.array([[True, True]])

    loss = mask_loss(mask_pred, mask_targets, positive_mask)

    expected_roi0 = math.log(2.0)  # class 0, target 1
    expected_roi1 = math.log1p(math.exp(-2.0))  # class 1, target 0
    expected = (expected_roi0 + expected_roi1) / 2.0

    np.testing.assert_allclose(loss, expected, rtol=1e-5)


def test_gradients_flow_to_positive_class_logits() -> None:
    mask_pred = jnp.array(
        [
            [
                [
                    [[-0.25]],
                    [[0.25]],
                ],
            ]
        ],
        dtype=jnp.float32,
    )
    mask_targets = _targets(
        masks=jnp.ones((1, 1, 1, 1), dtype=jnp.float32),
        classes=jnp.array([[1]], dtype=jnp.int32),
    )
    positive_mask = jnp.array([[True]])

    def loss_fn(logits: jnp.ndarray) -> jnp.ndarray:
        return mask_loss(logits, mask_targets, positive_mask)

    grads = jax.grad(loss_fn)(mask_pred)

    grad_class1 = grads[0, 0, 1, 0, 0]
    grad_class0 = grads[0, 0, 0, 0, 0]

    assert np.abs(np.asarray(grad_class1)) > 0.0
    np.testing.assert_allclose(np.asarray(grad_class0), 0.0, atol=1e-7)
