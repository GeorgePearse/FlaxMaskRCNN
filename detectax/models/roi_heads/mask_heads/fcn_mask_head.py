"""Fully convolutional mask head for Mask R-CNN in Flax."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float

BatchRoiFeatures = Float[Array, "batch num_rois in_height in_width in_channels"]
MaskLogits = Float[Array, "batch num_rois out_height out_width num_classes"]


class FCNMaskHead(nn.Module):
    """Mask prediction head based on stacked convolutions and upsampling.

    This implementation mirrors the Mask R-CNN mask head consisting of four
    3x3 convolutions followed by a 2x2 transposed convolution for upsampling
    and a final 1x1 convolution that produces per-class mask logits.

    Attributes:
        num_classes: Number of foreground classes to predict masks for.
        num_convs: Number of 3x3 convolutional layers applied before upsampling.
        conv_features: Number of channels produced by intermediate convolutions.
        use_bias: Whether to include bias parameters in convolutions.
        dtype: Computation dtype for all convolutions.
    """

    num_classes: int
    num_convs: int = 4
    conv_features: int = 256
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        roi_features: BatchRoiFeatures,
        *,
        train: bool = False,
    ) -> MaskLogits:
        """Apply the mask head to pooled RoI features.

        Args:
            roi_features: Region of Interest features with shape
                [batch, num_rois, 14, 14, in_channels].
            train: Unused flag for API parity with other modules.

        Returns:
            Per-class mask logits with shape
            [batch, num_rois, 28, 28, num_classes].
        """
        del train  # Currently unused but kept for interface compatibility.

        if roi_features.ndim != 5:
            raise ValueError(f"Expected 5D RoI features, got shape {roi_features.shape}")

        batch, num_rois, height, width, channels = roi_features.shape

        # Flatten RoI dimension into batch for convolutional processing.
        x = roi_features.reshape((batch * num_rois, height, width, channels))

        for i in range(self.num_convs):
            x = nn.Conv(
                features=self.conv_features,
                kernel_size=(3, 3),
                padding="SAME",
                use_bias=self.use_bias,
                dtype=self.dtype,
                name=f"mask_conv_{i}",
            )(x)
            x = nn.relu(x)

        x = nn.ConvTranspose(
            features=self.conv_features,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="SAME",
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="mask_deconv",
        )(x)
        x = nn.relu(x)

        logits = nn.Conv(
            features=self.num_classes,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=True,
            dtype=self.dtype,
            name="mask_logits",
        )(x)

        out_height, out_width = logits.shape[1:3]
        logits = logits.reshape((batch, num_rois, out_height, out_width, self.num_classes))
        return logits
