"""Feature Pyramid Network (FPN) implementation in Flax.

Reference: Feature Pyramid Networks for Object Detection
https://arxiv.org/abs/1612.03144

This implementation follows Scenic patterns for JAX/Flax while maintaining
architectural fidelity to the PyTorch visdet reference.
"""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class FPN(nn.Module):
    """Feature Pyramid Network.

    FPN builds a feature pyramid by combining high-resolution, semantically weak
    features with low-resolution, semantically strong features via lateral connections
    and top-down pathways.

    Attributes:
        in_channels: Number of input channels per scale (from backbone).
        out_channels: Number of output channels (used at each scale).
        num_outs: Number of output scales.
        start_level: Index of the start input backbone level. Default: 0.
        end_level: Index of the end input backbone level (exclusive).
                   Default: -1 (last level).
        add_extra_convs: Whether to add extra conv layers on top. Can be:
            - False: No extra convs
            - True/'on_input': Extra convs from last backbone feature
            - 'on_lateral': Extra convs from last lateral feature
            - 'on_output': Extra convs from last FPN output
        relu_before_extra_convs: Whether to apply relu before extra convs.
        use_bias: Whether to use bias in convolutions.
        dtype: Computation dtype (default: float32).

    Example:
        >>> fpn = FPN(
        ...     in_channels=[256, 512, 1024, 2048],
        ...     out_channels=256,
        ...     num_outs=5,
        ... )
        >>> # Inputs from backbone (e.g., ResNet C2, C3, C4, C5)
        >>> inputs = [
        ...     jnp.ones((2, 64, 64, 256)),   # C2
        ...     jnp.ones((2, 32, 32, 512)),   # C3
        ...     jnp.ones((2, 16, 16, 1024)),  # C4
        ...     jnp.ones((2, 8, 8, 2048)),    # C5
        ... ]
        >>> outputs = fpn.init_with_output(
        ...     jax.random.PRNGKey(0),
        ...     inputs,
        ... )[0]
        >>> len(outputs)  # 5 outputs (P2, P3, P4, P5, P6)
        5
        >>> [o.shape for o in outputs]
        [(2, 64, 64, 256), (2, 32, 32, 256), (2, 16, 16, 256),
         (2, 8, 8, 256), (2, 4, 4, 256)]
    """

    in_channels: Sequence[int]
    out_channels: int
    num_outs: int
    start_level: int = 0
    end_level: int = -1
    add_extra_convs: bool | str = False
    relu_before_extra_convs: bool = False
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        """Initialize FPN layers."""
        # Validate inputs
        assert isinstance(self.in_channels, (list, tuple))
        num_ins = len(self.in_channels)

        # Determine backbone end level
        if self.end_level == -1 or self.end_level == num_ins - 1:
            self.backbone_end_level = num_ins
            assert self.num_outs >= num_ins - self.start_level
        else:
            self.backbone_end_level = self.end_level + 1
            assert self.end_level < num_ins
            assert self.num_outs == self.end_level - self.start_level + 1

        # Validate add_extra_convs
        assert isinstance(self.add_extra_convs, (str, bool))
        if isinstance(self.add_extra_convs, str):
            assert self.add_extra_convs in ('on_input', 'on_lateral', 'on_output')

        # Store extra convs mode
        self.extra_convs_mode = self.add_extra_convs
        if self.add_extra_convs is True:
            self.extra_convs_mode = 'on_input'

    @nn.compact
    def __call__(
        self,
        inputs: Sequence[Float[Array, "batch height width channels"]],
        train: bool = False,
    ) -> Tuple[Float[Array, "batch height width out_channels"], ...]:
        """Forward pass of FPN.

        Args:
            inputs: Tuple of feature maps from backbone. Each is [B, H, W, C].
                   Typically (C2, C3, C4, C5) from ResNet.
            train: Whether in training mode.

        Returns:
            Tuple of FPN feature maps (P2, P3, P4, P5, P6, ...).
            Each is [B, H, W, out_channels].
        """
        assert len(inputs) == len(self.in_channels), (
            f"Expected {len(self.in_channels)} inputs, got {len(inputs)}"
        )

        # Build lateral connections (1x1 convs to unify channels)
        laterals = []
        for i in range(self.start_level, self.backbone_end_level):
            idx = i - self.start_level
            lateral = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                use_bias=self.use_bias,
                dtype=self.dtype,
                name=f'lateral_conv_{idx}',
            )(inputs[i])
            laterals.append(lateral)

        # Build top-down pathway (upsample and add)
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample higher-level feature
            prev_shape = laterals[i - 1].shape[1:3]  # [H, W]
            upsampled = jax.image.resize(
                laterals[i],
                shape=(laterals[i].shape[0], *prev_shape, laterals[i].shape[3]),
                method='nearest',
            )
            # Add to lateral connection
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply 3x3 conv to each lateral to get final FPN features
        outs = []
        for i, lateral in enumerate(laterals):
            fpn_out = nn.Conv(
                features=self.out_channels,
                kernel_size=(3, 3),
                padding='SAME',
                use_bias=self.use_bias,
                dtype=self.dtype,
                name=f'fpn_conv_{i}',
            )(lateral)
            outs.append(fpn_out)

        # Add extra levels if needed (for RetinaNet, etc.)
        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.extra_convs_mode and extra_levels >= 1:
            # Determine source for extra levels
            if self.extra_convs_mode == 'on_input':
                source = inputs[self.backbone_end_level - 1]
            elif self.extra_convs_mode == 'on_lateral':
                source = laterals[-1]
            else:  # 'on_output'
                source = outs[-1]

            # Add extra conv layers with stride 2
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_mode == 'on_input':
                    # First extra level from backbone feature
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                    extra_in = source
                else:
                    # Subsequent extra levels from previous output
                    in_channels = self.out_channels
                    extra_in = outs[-1]

                # Apply ReLU if requested
                if self.relu_before_extra_convs:
                    extra_in = nn.relu(extra_in)

                # 3x3 conv with stride 2 for downsampling
                extra_out = nn.Conv(
                    features=self.out_channels,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding='SAME',
                    use_bias=self.use_bias,
                    dtype=self.dtype,
                    name=f'extra_conv_{i}',
                )(extra_in)
                outs.append(extra_out)

        return tuple(outs)
