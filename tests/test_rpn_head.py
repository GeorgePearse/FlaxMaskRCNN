"""Unit tests for the Region Proposal Network head."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax import tree_util

from detectrax.models.heads.rpn_head import RPNHead
from detectrax.models.utils.anchor_generator import AnchorGenerator


def _build_fpn_features(
    batch_size: int,
    channels: int = 256,
) -> dict[str, jax.Array]:
    """Create synthetic FPN features for testing."""
    spatial_shapes = {
        "P2": (128, 128),
        "P3": (64, 64),
        "P4": (32, 32),
        "P5": (16, 16),
        "P6": (8, 8),
    }
    return {level: jnp.ones((batch_size, height, width, channels), dtype=jnp.float32) for level, (height, width) in spatial_shapes.items()}


class TestRPNHead:
    """Test-suite for :class:`~detectrax.models.heads.rpn_head.RPNHead`."""

    @pytest.fixture
    def rng_key(self) -> jax.Array:
        """PRNG key for module initialisation."""
        return jax.random.PRNGKey(0)

    def test_output_shapes(self, rng_key: jax.Array) -> None:
        """RPN head preserves spatial resolution and adjusts channel count."""
        features = _build_fpn_features(batch_size=2)
        head = RPNHead(num_anchors=3)

        variables = head.init(rng_key, features)
        objectness, deltas = head.apply(variables, features)

        for level, fmap in features.items():
            assert objectness[level].shape == fmap.shape[:3] + (head.num_anchors,)
            assert deltas[level].shape == fmap.shape[:3] + (head.num_anchors * 4,)

    def test_gradients_flow_through_branches(self, rng_key: jax.Array) -> None:
        """The gradients should propagate to both objectness and box heads."""
        features = _build_fpn_features(batch_size=1)
        head = RPNHead(num_anchors=3)
        variables = head.init(rng_key, features)

        def loss_fn(params: dict) -> jax.Array:
            objectness, deltas = head.apply({"params": params}, features)
            return jnp.sum(objectness["P3"]) + jnp.sum(deltas["P3"])

        grads = jax.grad(loss_fn)(variables["params"])
        assert jnp.any(jnp.abs(grads["objectness_conv"]["kernel"]) > 0)
        assert jnp.any(jnp.abs(grads["box_delta_conv"]["kernel"]) > 0)

    def test_objectness_bias_initialisation(self, rng_key: jax.Array) -> None:
        """Objectness bias should encode the requested prior probability."""
        head = RPNHead(num_anchors=3, prior_prob=0.01)
        variables = head.init(rng_key, _build_fpn_features(batch_size=1))
        bias = variables["params"]["objectness_conv"]["bias"]
        expected_bias = -math.log((1.0 - head.prior_prob) / head.prior_prob)
        assert jnp.allclose(bias, expected_bias)

    def test_multi_image_batching(self, rng_key: jax.Array) -> None:
        """Applying the same parameters to a larger batch should work."""
        head = RPNHead()
        features_batch1 = _build_fpn_features(batch_size=1)
        variables = head.init(rng_key, features_batch1)

        features_batch4 = _build_fpn_features(batch_size=4)
        objectness, deltas = head.apply(variables, features_batch4)

        assert objectness["P4"].shape[0] == 4
        assert deltas["P4"].shape[0] == 4

    def test_integration_with_anchor_generator(self, rng_key: jax.Array) -> None:
        """Anchors per level should align with predictions per spatial location."""
        head = RPNHead(num_anchors=3)
        features = _build_fpn_features(batch_size=1)
        variables = head.init(rng_key, features)
        objectness, _ = head.apply(variables, features)

        generator = AnchorGenerator()
        shapes = {level: fmap.shape[1:3] for level, fmap in features.items()}
        anchors = generator.generate(shapes, per_level=True)

        for level, fmap in features.items():
            height, width = fmap.shape[1:3]
            expected = height * width * generator.anchors_per_location
            assert anchors[level].shape[0] == expected
            assert objectness[level].shape[-1] == generator.anchors_per_location

    def test_parameter_count(self, rng_key: jax.Array) -> None:
        """Validate that the parameter tally matches the analytic expectation."""
        head = RPNHead(num_anchors=3)
        variables = head.init(rng_key, _build_fpn_features(batch_size=1))
        params = variables["params"]

        total_params = sum(int(p.size) for p in tree_util.tree_leaves(params))
        expected_total = (
            (3 * 3 * 256 * 256)
            + 256  # shared conv kernel + bias
            + (1 * 1 * 256 * head.num_anchors)
            + head.num_anchors  # objectness branch
            + (1 * 1 * 256 * head.num_anchors * 4)
            + (head.num_anchors * 4)  # box deltas
        )
        assert total_params == expected_total
