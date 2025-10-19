"""ResNet backbone wrapper for Dax.

Wraps Scenic's ResNet to extract multi-scale features for FPN.
"""

from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float

# We'll import from scenic_repo when needed
# For now, define interface


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction.

    This is a wrapper around Scenic's ResNet that extracts multi-scale
    features (C2, C3, C4, C5) for use with FPN.

    Attributes:
        num_layers: ResNet depth (18, 34, 50, 101, 152).
        num_filters: Base number of filters (default: 64).
        dtype: Computation dtype.

    Example:
        >>> backbone = ResNetBackbone(num_layers=50)
        >>> images = jnp.ones((2, 224, 224, 3))
        >>> features = backbone.init_with_output(
        ...     jax.random.PRNGKey(0),
        ...     images,
        ...     train=False,
        ... )[0]
        >>> # features is dict with keys: 'stage_1', 'stage_2', 'stage_3', 'stage_4'
        >>> # Corresponding to C2, C3, C4, C5 with strides 4, 8, 16, 32
    """

    num_layers: int = 50
    num_filters: int = 64
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch height width 3"],
        train: bool = False,
    ) -> Dict[str, Float[Array, "batch h w channels"]]:
        """Extract multi-scale features from input images.

        Args:
            x: Input images [B, H, W, 3].
            train: Whether in training mode (for BatchNorm).

        Returns:
            Dictionary of features:
            - 'stage_1' (C2): stride 4, channels 256 (ResNet-50)
            - 'stage_2' (C3): stride 8, channels 512
            - 'stage_3' (C4): stride 16, channels 1024
            - 'stage_4' (C5): stride 32, channels 2048
        """
        # TODO: Import and use Scenic's ResNet when available
        # For now, return placeholder features for testing
        batch_size = x.shape[0]
        h, w = x.shape[1], x.shape[2]

        # Simulate ResNet feature extraction
        # C2: stride 4, 256 channels
        c2 = nn.Conv(256, (1, 1), name='c2_placeholder')(x[:, ::4, ::4, :])
        # C3: stride 8, 512 channels
        c3 = nn.Conv(512, (1, 1), name='c3_placeholder')(x[:, ::8, ::8, :])
        # C4: stride 16, 1024 channels
        c4 = nn.Conv(1024, (1, 1), name='c4_placeholder')(x[:, ::16, ::16, :])
        # C5: stride 32, 2048 channels
        c5 = nn.Conv(2048, (1, 1), name='c5_placeholder')(x[:, ::32, ::32, :])

        return {
            'stage_1': c2,  # C2
            'stage_2': c3,  # C3
            'stage_3': c4,  # C4
            'stage_4': c5,  # C5
        }


def get_resnet_backbone(num_layers: int = 50) -> ResNetBackbone:
    """Factory function to create ResNet backbone.

    Args:
        num_layers: ResNet depth (18, 34, 50, 101, 152).

    Returns:
        ResNetBackbone module.
    """
    return ResNetBackbone(num_layers=num_layers)
