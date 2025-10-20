"""Tests for RPN proposal generation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from detectrax.models.task_modules.proposal_generator import generate_proposals


def _call_generator(
    anchors: jnp.ndarray,
    objectness: jnp.ndarray,
    deltas: jnp.ndarray,
    image_shape: jnp.ndarray,
    **kwargs: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return generate_proposals(
        anchors=anchors,
        objectness=objectness,
        deltas=deltas,
        image_shape=image_shape,
        **kwargs,
    )


def test_proposal_decoding_correctness() -> None:
    anchors = jnp.asarray(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray([[0.1, 0.9]], dtype=jnp.float32)
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray([[64.0, 64.0]], dtype=jnp.float32)

    proposals, scores = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=2,
        post_nms_top_n=2,
        nms_threshold=0.7,
        min_size=0.0,
    )

    expected_boxes = np.asarray(
        [
            [
                [20.0, 20.0, 30.0, 30.0],
                [0.0, 0.0, 10.0, 10.0],
            ]
        ],
        dtype=np.float32,
    )
    expected_scores = np.asarray([[0.9, 0.1]], dtype=np.float32)

    np.testing.assert_allclose(np.asarray(proposals), expected_boxes)
    np.testing.assert_allclose(np.asarray(scores), expected_scores)


def test_proposals_clipped_to_image_boundaries() -> None:
    anchors = jnp.asarray(
        [
            [
                [-5.0, -10.0, 60.0, 70.0],
            ]
        ],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray([[0.95]], dtype=jnp.float32)
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray([40.0, 50.0], dtype=jnp.float32)

    proposals, _ = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=1,
        post_nms_top_n=1,
    )

    expected = np.asarray([[[0.0, 0.0, 50.0, 40.0]]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(proposals), expected)


def test_small_box_filtering() -> None:
    anchors = jnp.asarray(
        [
            [
                [0.0, 0.0, 20.0, 20.0],
                [10.0, 10.0, 11.0, 11.0],
            ]
        ],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray([[0.9, 0.95]], dtype=jnp.float32)
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray([[32.0, 32.0]], dtype=jnp.float32)

    proposals, scores = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=2,
        post_nms_top_n=2,
        min_size=5.0,
    )

    np.testing.assert_allclose(np.asarray(scores)[0, 0], 0.9)
    np.testing.assert_allclose(np.asarray(scores)[0, 1], 0.0)
    np.testing.assert_allclose(np.asarray(proposals)[0, 1], np.zeros(4, dtype=np.float32))


def test_pre_nms_top_k_selection() -> None:
    anchors = jnp.asarray(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
                [40.0, 0.0, 50.0, 10.0],
                [60.0, 0.0, 70.0, 10.0],
                [80.0, 0.0, 90.0, 10.0],
            ]
        ],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray([[0.1, 0.2, 0.3, 0.9, 0.8]], dtype=jnp.float32)
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray([[128.0, 128.0]], dtype=jnp.float32)

    proposals, scores = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=3,
        post_nms_top_n=5,
    )

    expected_scores = np.asarray([0.9, 0.8, 0.3, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(scores)[0], expected_scores)


def test_nms_application_suppresses_overlaps() -> None:
    anchors = jnp.asarray(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 9.0, 9.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray([[0.9, 0.85, 0.8]], dtype=jnp.float32)
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray([[64.0, 64.0]], dtype=jnp.float32)

    proposals, scores = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=3,
        post_nms_top_n=3,
        nms_threshold=0.5,
    )

    np.testing.assert_allclose(np.asarray(scores)[0, :2], np.asarray([0.9, 0.8], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(scores)[0, 2], 0.0)

    kept_boxes = np.asarray(proposals)[0, :2]
    expected_boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(kept_boxes, expected_boxes)


def test_post_nms_top_k_selection_limits_results() -> None:
    anchors = jnp.asarray(
        [[[i * 10.0, 0.0, i * 10.0 + 5.0, 5.0] for i in range(6)]],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray([[0.95, 0.9, 0.85, 0.8, 0.75, 0.7]], dtype=jnp.float32)
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray([[64.0, 128.0]], dtype=jnp.float32)

    proposals, scores = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=6,
        post_nms_top_n=2,
    )

    assert proposals.shape == (1, 2, 4)
    assert scores.shape == (1, 2)
    np.testing.assert_allclose(np.asarray(scores)[0], np.asarray([0.95, 0.9], dtype=np.float32))


def test_batched_processing_generates_independent_results() -> None:
    anchors = jnp.asarray(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ],
            [
                [5.0, 5.0, 15.0, 15.0],
                [25.0, 25.0, 35.0, 35.0],
            ],
        ],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray(
        [
            [0.8, 0.3],
            [0.1, 0.95],
        ],
        dtype=jnp.float32,
    )
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray(
        [
            [64.0, 64.0],
            [32.0, 32.0],
        ],
        dtype=jnp.float32,
    )

    proposals, scores = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=2,
        post_nms_top_n=2,
    )

    np.testing.assert_allclose(np.asarray(scores)[0], np.asarray([0.8, 0.3], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(scores)[1], np.asarray([0.95, 0.1], dtype=np.float32))

    np.testing.assert_allclose(
        np.asarray(proposals)[0, 0],
        np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(proposals)[1, 0],
        np.asarray([25.0, 25.0, 32.0, 32.0], dtype=np.float32),
    )


def test_variable_input_sizes_result_in_padding() -> None:
    anchors = jnp.asarray(
        [
            [
                [0.0, 0.0, 12.0, 12.0],
                [15.0, 15.0, 17.0, 17.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 3.0, 3.0],
            ],
        ],
        dtype=jnp.float32,
    )
    objectness = jnp.asarray(
        [
            [0.6, 0.7],
            [0.8, 0.9],
        ],
        dtype=jnp.float32,
    )
    deltas = jnp.zeros_like(anchors)
    image_shape = jnp.asarray([[64.0, 64.0], [64.0, 64.0]], dtype=jnp.float32)

    proposals, scores = _call_generator(
        anchors,
        objectness,
        deltas,
        image_shape,
        pre_nms_top_n=2,
        post_nms_top_n=2,
        min_size=8.0,
    )

    # First image keeps one box, second filters all -> verify padding.
    np.testing.assert_allclose(np.asarray(scores)[0], np.asarray([0.6, 0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(scores)[1], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(proposals)[1], np.zeros((2, 4), dtype=np.float32))
