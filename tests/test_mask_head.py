"""Tests for the FCN mask head."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax.core import freeze, unfreeze

from detectax.models.roi_heads.mask_heads import FCNMaskHead


@pytest.fixture
def rng_key() -> jax.Array:
    """Provide a deterministic PRNG key."""
    return jax.random.PRNGKey(0)


def test_mask_head_output_shape(rng_key: jax.Array) -> None:
    """Mask head should upsample to 28x28 and return per-class logits."""
    mask_head = FCNMaskHead(num_classes=5)
    roi_features = jnp.ones((2, 3, 14, 14, 256), dtype=jnp.float32)

    variables = mask_head.init(rng_key, roi_features)
    outputs = mask_head.apply(variables, roi_features)

    assert outputs.shape == (2, 3, 28, 28, 5)


def test_mask_head_upsampling_identity(rng_key: jax.Array) -> None:
    """Transposed convolution should double resolution with stride 2."""
    mask_head = FCNMaskHead(num_classes=1, num_convs=0, conv_features=1)
    roi_features = jnp.zeros((1, 1, 14, 14, 1), dtype=jnp.float32)
    roi_features = roi_features.at[:, :, 2, 3, 0].set(1.0)

    variables = mask_head.init(rng_key, roi_features)
    params = unfreeze(variables["params"])

    # Configure transposed conv to copy values to (2*i, 2*j) positions.
    params["mask_deconv"]["kernel"] = jnp.zeros_like(params["mask_deconv"]["kernel"]).at[0, 0, 0, 0].set(1.0)
    params["mask_deconv"]["bias"] = jnp.zeros_like(params["mask_deconv"]["bias"])

    # Make final 1x1 conv a pass-through.
    params["mask_logits"]["kernel"] = jnp.zeros_like(params["mask_logits"]["kernel"]).at[0, 0, 0, 0].set(1.0)
    params["mask_logits"]["bias"] = jnp.zeros_like(params["mask_logits"]["bias"])

    outputs = mask_head.apply({"params": freeze(params)}, roi_features)

    expected = jnp.zeros((1, 1, 28, 28, 1), dtype=jnp.float32)
    # With `padding="SAME"` the non-zero voxel lands at (2*i+1, 2*j+1).
    expected = expected.at[:, :, 5, 7, 0].set(1.0)

    assert outputs.shape == expected.shape
    assert jnp.allclose(outputs, expected)


def test_mask_head_per_class_masks(rng_key: jax.Array) -> None:
    """Each class should have independent logits."""
    mask_head = FCNMaskHead(num_classes=2, num_convs=0, conv_features=1)
    roi_features = jnp.ones((1, 1, 14, 14, 1), dtype=jnp.float32)

    variables = mask_head.init(rng_key, roi_features)
    params = unfreeze(variables["params"])

    params["mask_deconv"]["kernel"] = jnp.zeros_like(params["mask_deconv"]["kernel"]).at[0, 0, 0, 0].set(1.0)
    params["mask_deconv"]["bias"] = jnp.zeros_like(params["mask_deconv"]["bias"])

    # Configure per-class weights differently.
    params["mask_logits"]["kernel"] = jnp.zeros_like(params["mask_logits"]["kernel"])
    params["mask_logits"]["kernel"] = params["mask_logits"]["kernel"].at[0, 0, 0, 0].set(1.0)
    params["mask_logits"]["kernel"] = params["mask_logits"]["kernel"].at[0, 0, 0, 1].set(2.0)
    params["mask_logits"]["bias"] = jnp.zeros_like(params["mask_logits"]["bias"])

    outputs = mask_head.apply({"params": freeze(params)}, roi_features)

    class0 = outputs[..., 0]
    class1 = outputs[..., 1]

    assert jnp.allclose(class1, 2.0 * class0)
    assert jnp.any(class0 != 0.0)


def test_mask_head_supports_gradients(rng_key: jax.Array) -> None:
    """Gradients should propagate through the mask head parameters."""
    mask_head = FCNMaskHead(num_classes=3)
    features_key, input_key = jax.random.split(rng_key)
    roi_features = jax.random.normal(input_key, (2, 4, 14, 14, 256))

    params = mask_head.init(features_key, roi_features)["params"]

    def loss_fn(p: dict[str, dict[str, jnp.ndarray]]) -> jnp.ndarray:
        logits = mask_head.apply({"params": p}, roi_features)
        return jnp.mean(logits)

    grads = jax.grad(loss_fn)(params)
    leaf_gradients = jax.tree_util.tree_leaves(grads)

    assert all(jnp.all(jnp.isfinite(g)) for g in leaf_gradients)
    assert any(jnp.any(g != 0.0) for g in leaf_gradients)
