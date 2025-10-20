"""Tests for mask post-processing utilities."""

from __future__ import annotations

import pytest

from detectrax.models.task_modules.post_processors import postprocess_masks

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_postprocess_masks_resizes_and_pastes_correctly() -> None:
    """Masks should resize to the ROI and paste into the full image."""
    mask_logits = jnp.zeros((1, 2, 28, 28), dtype=jnp.float32)
    mask_logits = mask_logits.at[0, 1, :, :].set(5.0)

    boxes = jnp.array([[2.0, 3.0, 8.0, 9.0]], dtype=jnp.float32)
    labels = jnp.array([1], dtype=jnp.int32)
    scores = jnp.array([0.95], dtype=jnp.float32)

    masks = postprocess_masks(
        mask_logits,
        {"boxes": boxes, "labels": labels, "scores": scores},
        image_shape=(12, 12),
    )

    assert masks.shape == (1, 12, 12)
    roi_region = masks[0, 3:9, 2:8]
    assert jnp.all(roi_region)
    outside = masks[0].at[3:9, 2:8].set(False)
    assert not bool(jnp.any(outside))


def test_postprocess_masks_applies_threshold() -> None:
    """Mask probabilities below the threshold should be suppressed."""
    mask_logits = jnp.full((1, 28, 28), -4.0, dtype=jnp.float32)
    boxes = jnp.array([[0.0, 0.0, 5.0, 5.0]], dtype=jnp.float32)

    masks = postprocess_masks(mask_logits, {"boxes": boxes}, image_shape=(6, 6), threshold=0.5)

    assert masks.shape == (1, 6, 6)
    assert not bool(jnp.any(masks))


def test_postprocess_masks_handles_overlapping_instances() -> None:
    """Higher-scoring instances should claim overlapping pixels."""
    mask_logits = jnp.full((2, 1, 28, 28), 4.0, dtype=jnp.float32)
    boxes = jnp.array(
        [
            [0.0, 0.0, 4.0, 4.0],
            [2.0, 2.0, 6.0, 6.0],
        ],
        dtype=jnp.float32,
    )
    labels = jnp.zeros((2,), dtype=jnp.int32)
    scores = jnp.array([0.4, 0.9], dtype=jnp.float32)

    masks = postprocess_masks(
        mask_logits,
        {"boxes": boxes, "labels": labels, "scores": scores},
        image_shape=(8, 8),
    )

    assert masks.shape == (2, 8, 8)

    overlap_slice = (slice(2, 4), slice(2, 4))
    assert not bool(jnp.any(masks[0][overlap_slice]))
    assert bool(jnp.all(masks[1][overlap_slice]))

    unique_region = (slice(0, 2), slice(0, 2))
    assert bool(jnp.all(masks[0][unique_region]))
