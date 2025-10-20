"""Tests for detection post-processing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from detectax.models.task_modules.post_processors import postprocess_detections


def _extract_single_detection_output(output):
    boxes, scores, labels = output
    assert len(boxes) == len(scores) == len(labels) == 1
    return boxes[0], scores[0], labels[0]


def test_score_threshold_filters_low_confidence() -> None:
    proposals = jnp.asarray(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ],
        dtype=jnp.float32,
    )
    cls_scores = jnp.asarray(
        [
            [
                [0.0, 3.0],
                [0.0, -3.0],
            ]
        ],
        dtype=jnp.float32,
    )
    box_deltas = jnp.zeros((1, 2, 8), dtype=jnp.float32)

    boxes, scores, labels = postprocess_detections(
        proposals,
        cls_scores,
        box_deltas,
        image_shape=(64, 64),
        score_threshold=0.5,
        nms_threshold=0.5,
        max_per_image=10,
    )

    single_boxes, single_scores, single_labels = _extract_single_detection_output((boxes, scores, labels))
    np.testing.assert_array_equal(single_boxes, np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(single_scores), np.asarray([0.95257413], dtype=np.float32), rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(single_labels), np.asarray([1], dtype=np.int32))


def test_per_class_nms_runs_independently() -> None:
    proposals = jnp.asarray([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]], dtype=jnp.float32)
    cls_scores = jnp.asarray(
        [
            [
                [0.0, 5.0, -5.0],  # Proposal 0 -> class 1
                [0.0, 4.5, 5.5],  # Proposal 1 -> classes 1 and 2
            ]
        ],
        dtype=jnp.float32,
    )
    box_deltas = jnp.zeros((1, 2, 12), dtype=jnp.float32)

    boxes, scores, labels = postprocess_detections(
        proposals,
        cls_scores,
        box_deltas,
        image_shape=(64, 64),
        score_threshold=0.05,
        nms_threshold=0.5,
        max_per_image=10,
    )

    det_boxes, det_scores, det_labels = _extract_single_detection_output((boxes, scores, labels))

    assert det_boxes.shape[0] == 2
    # Expect: class 1 box from proposal 0, class 2 box from proposal 1
    np.testing.assert_array_equal(det_labels, np.asarray([1, 2], dtype=np.int32))
    # Ensure NMS removed duplicate class 1 prediction from proposal 1
    expected_probs = jax.nn.softmax(cls_scores[0], axis=-1)
    np.testing.assert_allclose(det_scores[0], expected_probs[0, 1], rtol=1e-6)
    np.testing.assert_allclose(det_scores[1], expected_probs[1, 2], rtol=1e-6)
    np.testing.assert_array_equal(
        det_boxes,
        jnp.asarray(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
            ],
            dtype=jnp.float32,
        ),
    )


def test_top_k_selection_limits_detections() -> None:
    num_props = 5
    proposals = jnp.stack(
        [
            jnp.stack(
                [jnp.array([i * 5.0, i * 5.0, i * 5.0 + 4.0, i * 5.0 + 4.0], dtype=jnp.float32) for i in range(num_props)],
                axis=0,
            )
        ],
        axis=0,
    )
    logits = jnp.linspace(0.0, 4.0, num_props, dtype=jnp.float32)[None, :, None]
    cls_scores = jnp.concatenate([jnp.zeros_like(logits), logits], axis=-1)
    box_deltas = jnp.zeros((1, num_props, 8), dtype=jnp.float32)

    boxes, scores, labels = postprocess_detections(
        proposals,
        cls_scores,
        box_deltas,
        image_shape=(128, 128),
        score_threshold=0.0,
        nms_threshold=0.5,
        max_per_image=2,
    )

    det_boxes, det_scores, det_labels = _extract_single_detection_output((boxes, scores, labels))

    assert det_boxes.shape[0] == 2
    # Scores should be sorted descending by the postprocessor
    np.testing.assert_array_equal(det_labels, np.asarray([1, 1], dtype=np.int32))
    assert det_scores[0] >= det_scores[1]
    np.testing.assert_array_equal(
        det_boxes,
        proposals[0, -2:, :][::-1],
    )


def test_batch_processing_returns_per_image_outputs() -> None:
    proposals = jnp.asarray(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 28.0, 28.0],
            ],
            [
                [5.0, 5.0, 15.0, 15.0],
                [25.0, 25.0, 35.0, 35.0],
            ],
        ],
        dtype=jnp.float32,
    )
    cls_scores = jnp.asarray(
        [
            [
                [0.0, 2.0],
                [0.0, -2.0],
            ],
            [
                [0.0, -1.0],
                [0.0, 3.0],
            ],
        ],
        dtype=jnp.float32,
    )
    box_deltas = jnp.zeros((2, 2, 8), dtype=jnp.float32)
    image_shapes = jnp.asarray([[50, 50], [60, 60]], dtype=jnp.float32)

    boxes, scores, labels = postprocess_detections(
        proposals,
        cls_scores,
        box_deltas,
        image_shape=image_shapes,
        score_threshold=0.5,
        nms_threshold=0.5,
        max_per_image=5,
    )

    assert len(boxes) == 2
    assert len(scores) == 2
    assert len(labels) == 2

    probs = jax.nn.softmax(cls_scores, axis=-1)
    np.testing.assert_array_almost_equal(scores[0], np.asarray([probs[0, 0, 1]], dtype=np.float32))
    np.testing.assert_array_almost_equal(scores[1], np.asarray([probs[1, 1, 1]], dtype=np.float32))

    np.testing.assert_array_equal(labels[0], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(labels[1], np.asarray([1], dtype=np.int32))

    np.testing.assert_array_equal(boxes[0], np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32))
    np.testing.assert_array_equal(boxes[1], np.asarray([[25.0, 25.0, 35.0, 35.0]], dtype=np.float32))


@pytest.mark.parametrize(
    "bad_shape",
    [
        (64,),
        (32, 32, 32),
    ],
)
def test_invalid_image_shape_raises(bad_shape) -> None:
    proposals = jnp.zeros((1, 1, 4), dtype=jnp.float32)
    cls_scores = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    box_deltas = jnp.zeros((1, 1, 8), dtype=jnp.float32)

    with pytest.raises(ValueError):
        postprocess_detections(
            proposals,
            cls_scores,
            box_deltas,
            image_shape=bad_shape,
        )
