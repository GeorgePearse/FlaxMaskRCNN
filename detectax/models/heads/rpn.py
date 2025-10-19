"""Region Proposal Network (RPN) implementation in Flax.

Reference: Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
https://arxiv.org/abs/1506.01497
"""

from typing import List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class RPNHead(nn.Module):
    """Region Proposal Network Head.

    RPN predicts objectness scores and bounding box deltas for anchor boxes
    at each spatial location in the feature maps.

    Attributes:
        in_channels: Number of input channels from FPN.
        feat_channels: Number of channels in the intermediate conv layer.
        num_anchors: Number of anchors per spatial location (default: 3).
        dtype: Computation dtype.

    Example:
        >>> rpn = RPNHead(in_channels=256, feat_channels=256, num_anchors=3)
        >>> # Single feature map from FPN
        >>> feat = jnp.ones((2, 64, 64, 256))
        >>> cls_score, bbox_pred = rpn.init_with_output(
        ...     jax.random.PRNGKey(0),
        ...     feat,
        ... )[0]
        >>> cls_score.shape  # [B, H, W, num_anchors]
        (2, 64, 64, 3)
        >>> bbox_pred.shape  # [B, H, W, num_anchors * 4]
        (2, 64, 64, 12)
    """

    in_channels: int
    feat_channels: int = 256
    num_anchors: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "batch height width channels"],
        train: bool = False,
    ) -> Tuple[
        Float[Array, "batch height width num_anchors"],  # objectness scores
        Float[Array, "batch height width num_anchors*4"],  # bbox deltas
    ]:
        """Forward pass of RPN head.

        Args:
            x: Input feature map from FPN [B, H, W, C].
            train: Whether in training mode.

        Returns:
            Tuple of:
            - cls_score: Objectness scores [B, H, W, num_anchors]
            - bbox_pred: Bounding box deltas [B, H, W, num_anchors * 4]
        """
        # Shared 3x3 conv
        feat = nn.Conv(
            features=self.feat_channels,
            kernel_size=(3, 3),
            padding='SAME',
            dtype=self.dtype,
            name='rpn_conv',
        )(x)
        feat = nn.relu(feat)

        # Objectness classification (binary: object vs background)
        cls_score = nn.Conv(
            features=self.num_anchors,
            kernel_size=(1, 1),
            dtype=self.dtype,
            name='rpn_cls',
        )(feat)

        # Bounding box regression (4 values per anchor: dx, dy, dw, dh)
        bbox_pred = nn.Conv(
            features=self.num_anchors * 4,
            kernel_size=(1, 1),
            dtype=self.dtype,
            name='rpn_reg',
        )(feat)

        return cls_score, bbox_pred


class RPN(nn.Module):
    """Complete Region Proposal Network.

    Applies RPN head to multiple FPN levels and generates proposals.

    Attributes:
        in_channels: Number of input channels (same across all FPN levels).
        feat_channels: Number of intermediate channels.
        num_anchors: Number of anchors per location.
        dtype: Computation dtype.
    """

    in_channels: int = 256
    feat_channels: int = 256
    num_anchors: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        fpn_features: List[Float[Array, "batch height width channels"]],
        train: bool = False,
    ) -> Tuple[
        List[Float[Array, "batch height width num_anchors"]],
        List[Float[Array, "batch height width num_anchors*4"]],
    ]:
        """Apply RPN to all FPN levels.

        Args:
            fpn_features: List of feature maps from FPN (P2, P3, P4, P5, P6).
            train: Whether in training mode.

        Returns:
            Tuple of:
            - cls_scores: List of objectness scores for each FPN level
            - bbox_preds: List of bbox deltas for each FPN level
        """
        # Single RPN head shared across all FPN levels
        rpn_head = RPNHead(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            num_anchors=self.num_anchors,
            dtype=self.dtype,
            name='rpn_head',
        )

        cls_scores = []
        bbox_preds = []

        for feat in fpn_features:
            cls_score, bbox_pred = rpn_head(feat, train=train)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds
