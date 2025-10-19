"""Tests for the FPN anchor generator."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from detectax.models.utils import AnchorGenerator, generate_pyramid_anchors


@pytest.fixture
def generator() -> AnchorGenerator:
    """Default anchor generator fixture."""
    return AnchorGenerator()


def test_anchor_counts_per_level(generator: AnchorGenerator) -> None:
    """Each level should create H*W*num_anchors anchors."""
    shapes = {
        "P2": (4, 4),
        "P3": (2, 2),
        "P4": (1, 2),
        "P5": (1, 1),
        "P6": (1, 1),
    }
    anchors_by_level = generator.generate(shapes, per_level=True)
    anchors_per_location = generator.anchors_per_location

    for level, (height, width) in shapes.items():
        anchors = anchors_by_level[level]
        expected = height * width * anchors_per_location
        assert anchors.shape == (expected, 4), f"{level} expected {expected} anchors, got {anchors.shape}"


def test_anchor_dimensions_match_configuration(generator: AnchorGenerator) -> None:
    """Anchor widths and heights must reflect base sizes, scales, and ratios."""
    shapes = {level: (1, 1) for level in generator.level_names}
    anchors_p3 = generator.generate(shapes, per_level=True)["P3"]

    base_size = generator.base_sizes[1]
    stride = generator.strides[1]
    ratios = jnp.asarray(generator.aspect_ratios, dtype=jnp.float32)
    scales = jnp.asarray(generator.scales, dtype=jnp.float32)

    expected_widths = (base_size * (scales[:, None] / jnp.sqrt(ratios)[None, :])).reshape(-1)
    expected_heights = (base_size * (scales[:, None] * jnp.sqrt(ratios)[None, :])).reshape(-1)

    computed_widths = anchors_p3[:, 2] - anchors_p3[:, 0]
    computed_heights = anchors_p3[:, 3] - anchors_p3[:, 1]

    np.testing.assert_allclose(computed_widths, expected_widths, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(computed_heights, expected_heights, rtol=1e-5, atol=1e-5)

    centers_x = 0.5 * (anchors_p3[:, 0] + anchors_p3[:, 2])
    centers_y = 0.5 * (anchors_p3[:, 1] + anchors_p3[:, 3])
    expected_center = 0.5 * stride

    np.testing.assert_allclose(centers_x, expected_center)
    np.testing.assert_allclose(centers_y, expected_center)


def test_generate_returns_concatenated_anchors(generator: AnchorGenerator) -> None:
    """Combined anchor tensor should match the sum over levels."""
    shapes = {
        "P2": (2, 3),
        "P3": (1, 1),
        "P4": (2, 1),
        "P5": (1, 1),
        "P6": (1, 1),
    }
    anchors_by_level = generator.generate(shapes, per_level=True)
    concatenated = generator.generate(shapes, per_level=False)

    total_expected = sum(arr.shape[0] for arr in anchors_by_level.values())
    assert concatenated.shape == (total_expected, 4)

    # Validate ordering matches level_names concatenation.
    offset = 0
    for level in generator.level_names:
        level_anchors = anchors_by_level[level]
        level_count = level_anchors.shape[0]
        np.testing.assert_array_equal(concatenated[offset : offset + level_count], level_anchors)
        offset += level_count


def test_single_ratio_single_scale_edge_case() -> None:
    """Edge case with one ratio and one scale should still work."""
    generator = AnchorGenerator(
        aspect_ratios=(1.0,),
        scales=(0.5,),
        base_sizes=(16.0, 32.0, 64.0, 128.0, 256.0),
        strides=(4.0, 8.0, 16.0, 32.0, 64.0),
    )
    shapes = {level: (1, 1) for level in generator.level_names}
    anchors_by_level = generator.generate(shapes, per_level=True)

    for level, anchors in anchors_by_level.items():
        assert anchors.shape[0] == 1, f"{level} should have exactly one anchor."
        width = anchors[0, 2] - anchors[0, 0]
        height = anchors[0, 3] - anchors[0, 1]
        np.testing.assert_allclose(width, height)


def test_functional_api_matches_class(generator: AnchorGenerator) -> None:
    """Functional helper should mirror class-based generation."""
    shapes = [
        (4, 4),
        (2, 2),
        (1, 1),
        (1, 1),
        (1, 1),
    ]
    anchors_class = generator.generate(shapes, per_level=False)
    anchors_function = generate_pyramid_anchors(
        shapes,
        strides=generator.strides,
        base_sizes=generator.base_sizes,
        aspect_ratios=generator.aspect_ratios,
        scales=generator.scales,
        level_names=generator.level_names,
    )
    np.testing.assert_array_equal(anchors_class, anchors_function)
