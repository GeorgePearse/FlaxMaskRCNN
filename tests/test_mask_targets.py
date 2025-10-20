"""Tests for mask target generation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from pycocotools import mask as mask_utils

from detectax.models.task_modules import generate_mask_targets


def test_generate_mask_targets_crops_binary_mask() -> None:
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:6, 2:6] = 1.0

    positive_proposals = jnp.asarray([[2.0, 2.0, 6.0, 6.0]], dtype=jnp.float32)
    gt_boxes = jnp.asarray([[2.0, 2.0, 6.0, 6.0]], dtype=jnp.float32)

    targets = generate_mask_targets(positive_proposals, gt_boxes, [mask], mask_size=28)

    np.testing.assert_allclose(
        np.asarray(targets[0]),
        np.ones((28, 28), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_generate_mask_targets_uses_bilinear_resizing() -> None:
    mask = np.zeros((2, 2), dtype=np.float32)
    mask[:, 1] = 1.0

    positive_proposals = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)
    gt_boxes = jnp.asarray([[0.0, 0.0, 2.0, 2.0]], dtype=jnp.float32)

    targets = generate_mask_targets(positive_proposals, gt_boxes, [mask], mask_size=28)
    mask_target = np.asarray(targets[0])

    left_column_mean = float(mask_target[:, 0].mean())
    mid_column_mean = float(mask_target[:, mask_target.shape[1] // 2].mean())
    right_column_mean = float(mask_target[:, -1].mean())

    assert left_column_mean < 0.1
    np.testing.assert_allclose(mid_column_mean, 0.5, rtol=0.1, atol=0.1)
    assert right_column_mean > 0.9


def test_generate_mask_targets_supports_polygon_masks() -> None:
    polygon = [[2.0, 2.0, 7.0, 2.0, 7.0, 7.0, 2.0, 7.0]]
    polygon_annotation = {
        "polygons": polygon,
        "height": 10,
        "width": 10,
    }

    positive_proposals = jnp.asarray([[2.0, 2.0, 7.0, 7.0]], dtype=jnp.float32)
    gt_boxes = jnp.asarray([[2.0, 2.0, 7.0, 7.0]], dtype=jnp.float32)

    targets = generate_mask_targets(positive_proposals, gt_boxes, [polygon_annotation], mask_size=28)

    np.testing.assert_allclose(
        np.asarray(targets[0]),
        np.ones((28, 28), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_generate_mask_targets_supports_rle_masks() -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:9, 4:9] = 1

    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")

    positive_proposals = jnp.asarray([[4.0, 4.0, 9.0, 9.0]], dtype=jnp.float32)
    gt_boxes = jnp.asarray([[4.0, 4.0, 9.0, 9.0]], dtype=jnp.float32)

    targets = generate_mask_targets(positive_proposals, gt_boxes, [rle], mask_size=28)

    np.testing.assert_allclose(
        np.asarray(targets[0]),
        np.ones((28, 28), dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
