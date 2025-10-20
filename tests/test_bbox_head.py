"""Unit tests for the second-stage bounding-box head."""

from __future__ import annotations

import pytest

from detectax.models.layers.roi_align import roi_align
from detectax.models.roi_heads.bbox_heads import BBoxHead

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class TestBBoxHead:
    """Comprehensive test-suite for :class:`~detectax.models.roi_heads.bbox_heads.BBoxHead`."""

    @pytest.fixture
    def rng_key(self) -> jax.Array:
        """Deterministic PRNG key for test reproducibility."""
        return jax.random.PRNGKey(0)

    def test_output_shapes(self, rng_key: jax.Array) -> None:
        """Head produces logits and box deltas with expected shapes."""
        num_classes = 5
        head = BBoxHead(num_classes=num_classes)
        roi_features = jnp.ones((2, 8, 7, 7, 256), dtype=jnp.float32)

        variables = head.init(rng_key, roi_features)
        cls_logits, bbox_deltas = head.apply(variables, roi_features)

        assert cls_logits.shape == (2, 8, num_classes)
        assert bbox_deltas.shape == (2, 8, num_classes * 4)

    def test_gradients_flow(self, rng_key: jax.Array) -> None:
        """Gradients must propagate through both classification and regression heads."""
        head = BBoxHead(num_classes=4)
        roi_features = jnp.ones((1, 6, 7, 7, 256), dtype=jnp.float32)
        variables = head.init(rng_key, roi_features)

        def loss_fn(features: jax.Array) -> jax.Array:
            cls_logits, bbox_deltas = head.apply(variables, features)
            return jnp.sum(cls_logits) + jnp.sum(bbox_deltas)

        grads = jax.grad(loss_fn)(roi_features)
        assert grads.shape == roi_features.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_class_specific_vs_agnostic(self, rng_key: jax.Array) -> None:
        """Regression branch should switch dimensionality based on the configuration."""
        roi_features = jnp.ones((1, 4, 7, 7, 256), dtype=jnp.float32)
        key_specific, key_agnostic = jax.random.split(rng_key)

        specific_head = BBoxHead(num_classes=3, class_agnostic=False)
        specific_vars = specific_head.init(key_specific, roi_features)
        _, specific_deltas = specific_head.apply(specific_vars, roi_features)
        assert specific_deltas.shape == (1, 4, 12)  # 3 classes * 4 deltas

        agnostic_head = BBoxHead(num_classes=3, class_agnostic=True)
        agnostic_vars = agnostic_head.init(key_agnostic, roi_features)
        _, agnostic_deltas = agnostic_head.apply(agnostic_vars, roi_features)
        assert agnostic_deltas.shape == (1, 4, 4)

    def test_regression_head_initialisation_std(self, rng_key: jax.Array) -> None:
        """Regression branch must use a small weight initialisation."""
        head = BBoxHead(num_classes=2)
        roi_features = jnp.ones((1, 1, 7, 7, 256), dtype=jnp.float32)
        variables = head.init(rng_key, roi_features)

        kernel = variables["params"]["bbox_deltas"]["kernel"]
        observed_std = jnp.std(kernel)

        assert jnp.isfinite(observed_std)
        assert float(observed_std) == pytest.approx(0.001, rel=0.25)

    def test_integration_with_roi_align(self, rng_key: jax.Array) -> None:
        """Head operates on features generated via RoI Align from an FPN level."""
        feature_map = jnp.linspace(0.0, 1.0, num=16 * 16 * 256, dtype=jnp.float32).reshape((1, 16, 16, 256))
        boxes = jnp.array(
            [
                [1.0, 2.0, 10.0, 14.0],
                [3.0, 5.0, 12.0, 12.0],
                [4.0, 4.0, 15.0, 15.0],
            ],
            dtype=jnp.float32,
        )

        aligned = roi_align(feature_map, boxes, output_size=(7, 7), spatial_scale=1.0, sampling_ratio=1)
        batched_rois = aligned[jnp.newaxis, ...]  # [1, num_rois, 7, 7, 256]

        head = BBoxHead(num_classes=4)
        variables = head.init(rng_key, batched_rois)
        cls_logits, bbox_deltas = head.apply(variables, batched_rois)

        assert cls_logits.shape == (1, boxes.shape[0], 4)
        assert bbox_deltas.shape == (1, boxes.shape[0], 16)
        assert jnp.all(jnp.isfinite(cls_logits))
        assert jnp.all(jnp.isfinite(bbox_deltas))
