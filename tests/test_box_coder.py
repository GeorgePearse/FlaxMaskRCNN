"""Tests for bounding box encoding and decoding utilities."""

from __future__ import annotations

import numpy as np
import pytest

from detectrax.models.utils import decode_boxes, encode_boxes

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_round_trip_encode_decode_identity() -> None:
    """Encoding followed by decoding should reproduce the original boxes."""
    anchors = jnp.array(
        [
            [10.0, 15.0, 30.0, 45.0],
            [20.0, 25.0, 50.0, 65.0],
            [5.0, 5.0, 25.0, 35.0],
        ],
        dtype=jnp.float32,
    )
    boxes = anchors + jnp.array(
        [
            [-2.0, 1.5, 4.0, -3.0],
            [1.0, -2.0, -1.0, 3.0],
            [0.5, -0.5, 0.0, 2.0],
        ],
        dtype=jnp.float32,
    )

    deltas = encode_boxes(boxes, anchors)
    decoded = decode_boxes(deltas, anchors)

    np.testing.assert_allclose(decoded, boxes, rtol=1e-5, atol=1e-5)


def test_gradients_flow_through_encode_decode() -> None:
    """Gradients should propagate through the encode/decode pipeline."""
    anchors = jnp.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 20.0, 18.0],
        ],
        dtype=jnp.float32,
    )
    boxes = anchors + jnp.array(
        [
            [1.0, -1.0, 2.0, 0.5],
            [-0.5, 0.25, 1.5, -0.75],
        ],
        dtype=jnp.float32,
    )

    def loss_fn(b: jnp.ndarray) -> jnp.ndarray:
        deltas = encode_boxes(b, anchors)
        decoded = decode_boxes(deltas, anchors)
        return jnp.sum(decoded * b)

    grads = jax.grad(loss_fn)(boxes)
    assert jnp.all(jnp.isfinite(grads)), "Gradient contains non-finite values."
    assert grads.shape == boxes.shape


def test_batch_processing_with_vmap() -> None:
    """Batch inputs mapped with :func:`jax.vmap` should operate correctly."""
    anchors = jnp.array(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 15.0, 35.0, 30.0],
            ],
            [
                [5.0, 5.0, 15.0, 20.0],
                [12.0, 10.0, 28.0, 26.0],
            ],
        ],
        dtype=jnp.float32,
    )
    boxes = anchors + jnp.array(
        [
            [
                [1.0, -1.0, 2.5, -0.5],
                [-0.5, 2.0, 1.0, 3.5],
            ],
            [
                [0.25, -0.25, 1.0, 0.5],
                [1.5, -1.0, 2.0, 1.0],
            ],
        ],
        dtype=jnp.float32,
    )

    vmapped_encode = jax.vmap(encode_boxes, in_axes=(0, 0))
    vmapped_decode = jax.vmap(decode_boxes, in_axes=(0, 0))

    deltas = vmapped_encode(boxes, anchors)
    decoded = vmapped_decode(deltas, anchors)

    np.testing.assert_allclose(decoded, boxes, rtol=1e-5, atol=1e-5)


def test_zero_size_boxes_are_handled() -> None:
    """Zero-size boxes should not introduce NaNs or infs."""
    anchors = jnp.array(
        [
            [10.0, 10.0, 20.0, 20.0],
            [5.0, 5.0, 5.0, 12.0],  # zero-width anchor
        ],
        dtype=jnp.float32,
    )
    boxes = jnp.array(
        [
            [12.0, 12.0, 18.0, 18.0],
            [5.0, 5.0, 5.0, 5.0],  # zero-size gt box
        ],
        dtype=jnp.float32,
    )

    deltas = encode_boxes(boxes, anchors)
    decoded = decode_boxes(deltas, anchors)

    assert jnp.all(jnp.isfinite(deltas)), "Encoding produced non-finite values."
    assert jnp.all(jnp.isfinite(decoded)), "Decoding produced non-finite values."


@pytest.mark.parametrize("scale", [5.0, 10.0])
def test_decode_clipping_behavior(scale: float) -> None:
    """Decoded widths/heights should be capped by the clipping threshold."""
    anchors = jnp.array([[0.0, 0.0, 10.0, 20.0]], dtype=jnp.float32)
    extreme_deltas = jnp.array([[0.0, 0.0, scale, scale]], dtype=jnp.float32)
    clip = 4.135

    decoded = decode_boxes(extreme_deltas, anchors, clip_value=clip)

    expected_width = (anchors[0, 2] - anchors[0, 0]) * jnp.exp(clip)
    expected_height = (anchors[0, 3] - anchors[0, 1]) * jnp.exp(clip)

    width = decoded[0, 2] - decoded[0, 0]
    height = decoded[0, 3] - decoded[0, 1]

    np.testing.assert_allclose(width, expected_width, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(height, expected_height, rtol=1e-5, atol=1e-5)
