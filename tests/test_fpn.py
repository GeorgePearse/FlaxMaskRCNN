"""Unit tests for Feature Pyramid Network (FPN)."""

import jax
import jax.numpy as jnp
import pytest

from detectrax.models.necks.fpn import FPN


class TestFPN:
    """Test suite for FPN module."""

    @pytest.fixture
    def rng_key(self) -> jax.Array:
        """Provide a PRNG key for tests."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def resnet_features(self) -> tuple[jax.Array, ...]:
        """Simulated ResNet-50 output features (C2, C3, C4, C5).

        Returns:
            Tuple of 4 feature maps with shapes:
            - C2: [2, 64, 64, 256]   (stride 4)
            - C3: [2, 32, 32, 512]   (stride 8)
            - C4: [2, 16, 16, 1024]  (stride 16)
            - C5: [2, 8, 8, 2048]    (stride 32)
        """
        return (
            jnp.ones((2, 64, 64, 256)),
            jnp.ones((2, 32, 32, 512)),
            jnp.ones((2, 16, 16, 1024)),
            jnp.ones((2, 8, 8, 2048)),
        )

    def test_fpn_basic_initialization(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test basic FPN initialization and forward pass."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        variables = fpn.init(rng_key, resnet_features)
        outputs = fpn.apply(variables, resnet_features)

        # Check number of outputs
        assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

        # Check output shapes
        expected_shapes = [
            (2, 64, 64, 256),  # P2
            (2, 32, 32, 256),  # P3
            (2, 16, 16, 256),  # P4
            (2, 8, 8, 256),  # P5
        ]
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, f"P{i + 2} shape mismatch: expected {expected_shape}, got {output.shape}"

        # Check all outputs have same channel dimension
        for i, output in enumerate(outputs):
            assert output.shape[-1] == 256, f"P{i + 2} channels mismatch: expected 256, got {output.shape[-1]}"

    def test_fpn_with_extra_levels(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN with extra pyramid levels (e.g., P6)."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,  # P2, P3, P4, P5, P6
            add_extra_convs="on_input",
        )

        variables = fpn.init(rng_key, resnet_features)
        outputs = fpn.apply(variables, resnet_features)

        # Check number of outputs
        assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"

        # Check output shapes (including P6)
        expected_shapes = [
            (2, 64, 64, 256),  # P2
            (2, 32, 32, 256),  # P3
            (2, 16, 16, 256),  # P4
            (2, 8, 8, 256),  # P5
            (2, 4, 4, 256),  # P6 (stride 64, downsampled from P5)
        ]
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, f"P{i + 2} shape mismatch: expected {expected_shape}, got {output.shape}"

    def test_fpn_extra_convs_on_lateral(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN with extra convs from lateral features."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            add_extra_convs="on_lateral",
        )

        variables = fpn.init(rng_key, resnet_features)
        outputs = fpn.apply(variables, resnet_features)

        assert len(outputs) == 5
        assert outputs[-1].shape == (2, 4, 4, 256), "P6 should be downsampled"

    def test_fpn_extra_convs_on_output(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN with extra convs from output features."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            add_extra_convs="on_output",
        )

        variables = fpn.init(rng_key, resnet_features)
        outputs = fpn.apply(variables, resnet_features)

        assert len(outputs) == 5
        assert outputs[-1].shape == (2, 4, 4, 256)

    def test_fpn_with_start_end_level(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN with custom start and end levels."""
        # Use only C3, C4, C5 (skip C2)
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=3,
            start_level=1,  # Start from C3
            end_level=3,  # End at C5 (inclusive)
        )

        variables = fpn.init(rng_key, resnet_features)
        outputs = fpn.apply(variables, resnet_features)

        # Should output P3, P4, P5
        assert len(outputs) == 3
        expected_shapes = [
            (2, 32, 32, 256),  # P3
            (2, 16, 16, 256),  # P4
            (2, 8, 8, 256),  # P5
        ]
        for output, expected_shape in zip(outputs, expected_shapes):
            assert output.shape == expected_shape

    def test_fpn_different_out_channels(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN with different output channel dimension."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=512,  # Different from standard 256
            num_outs=4,
        )

        variables = fpn.init(rng_key, resnet_features)
        outputs = fpn.apply(variables, resnet_features)

        # All outputs should have 512 channels
        for output in outputs:
            assert output.shape[-1] == 512, f"Expected 512 channels, got {output.shape[-1]}"

    def test_fpn_batch_size_invariance(self, rng_key: jax.Array) -> None:
        """Test FPN works with different batch sizes."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        # Test with batch size 1
        inputs_batch1 = (
            jnp.ones((1, 64, 64, 256)),
            jnp.ones((1, 32, 32, 512)),
            jnp.ones((1, 16, 16, 1024)),
            jnp.ones((1, 8, 8, 2048)),
        )
        variables = fpn.init(rng_key, inputs_batch1)
        outputs_batch1 = fpn.apply(variables, inputs_batch1)

        assert outputs_batch1[0].shape[0] == 1

        # Test with batch size 8 using same parameters
        inputs_batch8 = (
            jnp.ones((8, 64, 64, 256)),
            jnp.ones((8, 32, 32, 512)),
            jnp.ones((8, 16, 16, 1024)),
            jnp.ones((8, 8, 8, 2048)),
        )
        outputs_batch8 = fpn.apply(variables, inputs_batch8)

        assert outputs_batch8[0].shape[0] == 8

    def test_fpn_deterministic_output(self, resnet_features: tuple) -> None:
        """Test FPN produces same output with same key."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        key = jax.random.PRNGKey(42)
        variables = fpn.init(key, resnet_features)

        # Run forward pass twice
        outputs1 = fpn.apply(variables, resnet_features)
        outputs2 = fpn.apply(variables, resnet_features)

        # Outputs should be identical
        for o1, o2 in zip(outputs1, outputs2):
            assert jnp.allclose(o1, o2), "FPN should be deterministic"

    def test_fpn_parameter_count(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN has reasonable number of parameters."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        variables = fpn.init(rng_key, resnet_features)
        params = variables["params"]

        # Count total parameters
        total_params = sum(p.size for p in jax.tree.leaves(params))

        # FPN should have a reasonable number of params
        # Rough estimate: 4 lateral 1x1 convs + 4 FPN 3x3 convs
        # Each 1x1: in_ch * 256
        # Each 3x3: 256 * 3 * 3 * 256
        min_params = 10_000
        max_params = 10_000_000

        assert min_params < total_params < max_params, f"Parameter count {total_params} seems unreasonable"

    def test_fpn_invalid_config_raises(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN raises errors for invalid configurations."""
        # Test invalid add_extra_convs
        with pytest.raises(AssertionError):
            fpn = FPN(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
                add_extra_convs="invalid_mode",  # type: ignore
            )
            fpn.init(rng_key, resnet_features)

    def test_fpn_gradient_flow(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test gradients can flow through FPN."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        def loss_fn(params, inputs):
            outputs = fpn.apply({"params": params}, inputs)
            # Simple loss: sum of all outputs
            return sum(jnp.sum(o) for o in outputs)

        variables = fpn.init(rng_key, resnet_features)
        params = variables["params"]

        # Compute gradients
        grads = jax.grad(loss_fn)(params, resnet_features)

        # Check gradients exist and are not zero
        for grad_leaf in jax.tree.leaves(grads):
            assert jnp.any(grad_leaf != 0), "Gradients should be non-zero"

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_fpn_various_batch_sizes(
        self,
        rng_key: jax.Array,
        batch_size: int,
    ) -> None:
        """Test FPN with various batch sizes."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        inputs = (
            jnp.ones((batch_size, 64, 64, 256)),
            jnp.ones((batch_size, 32, 32, 512)),
            jnp.ones((batch_size, 16, 16, 1024)),
            jnp.ones((batch_size, 8, 8, 2048)),
        )

        variables = fpn.init(rng_key, inputs)
        outputs = fpn.apply(variables, inputs)

        for output in outputs:
            assert output.shape[0] == batch_size

    def test_fpn_jit_compilation(self, rng_key: jax.Array, resnet_features: tuple) -> None:
        """Test FPN can be JIT compiled."""
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=4,
        )

        variables = fpn.init(rng_key, resnet_features)

        # JIT compile the forward pass
        @jax.jit
        def forward(vars, inputs):
            return fpn.apply(vars, inputs)

        # Run JIT compiled version
        outputs = forward(variables, resnet_features)

        assert len(outputs) == 4
        for output in outputs:
            assert output.shape[-1] == 256
