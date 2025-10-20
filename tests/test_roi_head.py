"""Tests for the RoI head base class."""

import math

import jax
import jax.numpy as jnp
import pytest

from detectrax.models.roi_heads import BaseRoIHead


@pytest.fixture(autouse=True)
def fast_roi_align(monkeypatch):
    """Patch RoI Align with a lightweight deterministic stub for tests."""

    def _stub_roi_align(features, boxes, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=2):
        level = int(round(math.log2(1.0 / spatial_scale)))
        value = float(level)
        return jnp.full((boxes.shape[0], output_size[0], output_size[1], features.shape[-1]), value, dtype=features.dtype)

    monkeypatch.setattr(
        "detectrax.models.roi_heads.base_roi_head.roi_align",
        _stub_roi_align,
    )


class DummyRoIHead(BaseRoIHead):
    """Minimal concrete implementation used for testing."""

    def box_head(self, roi_features, train: bool = False):  # type: ignore[override]
        return jnp.mean(roi_features, axis=(-3, -2))

    def get_targets(self, proposals, gt_boxes, gt_labels):  # type: ignore[override]
        return {"proposals": proposals, "gt_boxes": gt_boxes, "gt_labels": gt_labels}

    def get_predictions(self, box_features):  # type: ignore[override]
        return {"box_features": box_features}

    def loss(self, *args, **kwargs):  # type: ignore[override]
        return {}


def _make_fpn_features(batch_size: int = 1, channels: int = 16):
    """Create synthetic FPN feature maps."""

    level_sizes = {"p2": 64, "p3": 32, "p4": 16, "p5": 8, "p6": 4}
    features = {}
    for level, spatial in level_sizes.items():
        template = jnp.zeros((spatial, spatial, channels))
        stacked = jnp.stack([template for _ in range(batch_size)])
        features[level] = stacked
    return features


def _run_extract(head: DummyRoIHead, features, proposals):
    variables = head.init(
        jax.random.PRNGKey(0),
        features,
        proposals,
        method=head.extract_roi_features,
    )
    return head.apply(
        variables,
        features,
        proposals,
        method=head.extract_roi_features,
    )


def test_roi_extraction_shape():
    head = DummyRoIHead(roi_output_size=(14, 14))
    features = _make_fpn_features(batch_size=1)
    proposals = jnp.array(
        [
            [
                [0.0, 0.0, 64.0, 64.0],
                [32.0, 32.0, 160.0, 224.0],
            ]
        ]
    )

    roi_feats = _run_extract(head, features, proposals)

    assert roi_feats.shape == (1, 2, 14, 14, 16)


def test_multi_scale_level_assignment():
    head = DummyRoIHead()
    features = _make_fpn_features(batch_size=1)

    proposals = jnp.array(
        [
            [
                [0.0, 0.0, 32.0, 32.0],  # Small box -> expect P2 (value 2)
                [0.0, 0.0, 160.0, 160.0],  # Medium box -> expect P3 (value 3)
                [0.0, 0.0, 1024.0, 1024.0],  # Very large box -> expect P6 (value 6)
            ]
        ]
    )

    roi_feats = _run_extract(head, features, proposals)

    means = jnp.mean(roi_feats[0], axis=(1, 2, 3))
    assert float(means[0]) == pytest.approx(2.0, rel=1e-5)
    assert float(means[1]) == pytest.approx(3.0, rel=1e-5)
    assert float(means[2]) == pytest.approx(6.0, rel=1e-5)


def test_batch_processing_per_image_features():
    head = DummyRoIHead()
    features = _make_fpn_features(batch_size=2)

    proposals = jnp.array(
        [
            [[0.0, 0.0, 48.0, 48.0]],  # Should map to P2 -> level value 2
            [[0.0, 0.0, 256.0, 256.0]],  # Should map to P4 -> level value 4
        ]
    )

    roi_feats = _run_extract(head, features, proposals)

    first_mean = float(jnp.mean(roi_feats[0, 0]))
    second_mean = float(jnp.mean(roi_feats[1, 0]))

    assert first_mean == pytest.approx(2.0, rel=1e-5)
    assert second_mean == pytest.approx(4.0, rel=1e-5)


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        BaseRoIHead()  # type: ignore[abstract]
