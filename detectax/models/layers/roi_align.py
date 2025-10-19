"""RoI Align operation in JAX.

Reference: Mask R-CNN
https://arxiv.org/abs/1703.06870
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def roi_align(
    features: Float[Array, "batch height width channels"],
    boxes: Float[Array, "num_boxes 4"],
    output_size: tuple[int, int] = (7, 7),
    spatial_scale: float = 1.0,
    sampling_ratio: int = 2,
) -> Float[Array, "num_boxes out_h out_w channels"]:
    """RoI Align operation.

    Extracts fixed-size feature maps from regions of interest using
    bilinear interpolation for sub-pixel alignment.

    Args:
        features: Input feature map [B, H, W, C].
        boxes: Region of interest boxes [N, 4] in (x1, y1, x2, y2) format.
               Coordinates are in original image space.
        output_size: Output spatial size (height, width). Default: (7, 7).
        spatial_scale: Scale factor to map boxes to feature map coordinates.
                       For FPN P2 (stride 4), use 1/4 = 0.25.
        sampling_ratio: Number of sampling points per bin. Default: 2.
                       -1 means adaptive (use bin size).

    Returns:
        Aligned features [N, out_h, out_w, C].

    Example:
        >>> features = jnp.ones((1, 64, 64, 256))
        >>> boxes = jnp.array([[10.0, 10.0, 50.0, 50.0]])  # In image coords
        >>> aligned = roi_align(features, boxes, output_size=(7, 7), spatial_scale=1/4.0)
        >>> aligned.shape
        (1, 7, 7, 256)
    """
    _batch_size, feat_h, feat_w, channels = features.shape
    _num_boxes = boxes.shape[0]
    out_h, out_w = output_size

    # Scale boxes to feature map coordinates
    scaled_boxes = boxes * spatial_scale

    # For now, assume batch_size = 1 for simplicity
    # TODO: Handle multi-batch by adding batch indices to boxes
    feat_single = features[0]  # [H, W, C]

    def align_single_box(box: Float[Array, "4"]) -> Float[Array, "out_h out_w channels"]:
        """Align a single box."""
        x1, y1, x2, y2 = box

        # Calculate bin size
        roi_h = jnp.maximum(y2 - y1, 1.0)
        roi_w = jnp.maximum(x2 - x1, 1.0)
        bin_h = roi_h / out_h
        bin_w = roi_w / out_w

        # Determine sampling points per bin
        if sampling_ratio > 0:
            roi_bin_grid_h = sampling_ratio
            roi_bin_grid_w = sampling_ratio
        else:
            # Adaptive: ceil(bin_size)
            roi_bin_grid_h = jnp.ceil(bin_h).astype(jnp.int32)
            roi_bin_grid_w = jnp.ceil(bin_w).astype(jnp.int32)

        # Generate sampling grid
        def sample_bin(iy: int, ix: int) -> Float[Array, "channels"]:
            """Sample a single bin using bilinear interpolation."""
            # Bin center in RoI coordinates
            y_center = y1 + (iy + 0.5) * bin_h
            x_center = x1 + (ix + 0.5) * bin_w

            # Sample points within the bin
            samples = []
            for py in range(roi_bin_grid_h):
                for px in range(roi_bin_grid_w):
                    # Sampling point in feature map coordinates
                    y = y_center + (py + 0.5) * bin_h / roi_bin_grid_h - 0.5
                    x = x_center + (px + 0.5) * bin_w / roi_bin_grid_w - 0.5

                    # Bilinear interpolation
                    x = jnp.clip(x, 0, feat_w - 1)
                    y = jnp.clip(y, 0, feat_h - 1)

                    x0 = jnp.floor(x).astype(jnp.int32)
                    y0 = jnp.floor(y).astype(jnp.int32)
                    x1_pt = jnp.minimum(x0 + 1, feat_w - 1)
                    y1_pt = jnp.minimum(y0 + 1, feat_h - 1)

                    # Interpolation weights
                    wx1 = x - x0
                    wx0 = 1.0 - wx1
                    wy1 = y - y0
                    wy0 = 1.0 - wy1

                    # Gather values
                    v00 = feat_single[y0, x0, :]
                    v01 = feat_single[y0, x1_pt, :]
                    v10 = feat_single[y1_pt, x0, :]
                    v11 = feat_single[y1_pt, x1_pt, :]

                    # Bilinear interpolation
                    val = wy0 * wx0 * v00 + wy0 * wx1 * v01 + wy1 * wx0 * v10 + wy1 * wx1 * v11
                    samples.append(val)

            # Average all samples in the bin
            return jnp.mean(jnp.stack(samples, axis=0), axis=0)

        # Vectorize over output grid
        output = jnp.zeros((out_h, out_w, channels))
        for iy in range(out_h):
            for ix in range(out_w):
                output = output.at[iy, ix].set(sample_bin(iy, ix))

        return output

    # Apply to all boxes
    aligned_features = jax.vmap(align_single_box)(scaled_boxes)

    return aligned_features


def roi_pool(
    features: Float[Array, "batch height width channels"],
    boxes: Float[Array, "num_boxes 4"],
    output_size: tuple[int, int] = (7, 7),
    spatial_scale: float = 1.0,
) -> Float[Array, "num_boxes out_h out_w channels"]:
    """RoI Pooling operation (simpler alternative to RoI Align).

    Uses max pooling instead of bilinear interpolation. Faster but less accurate.

    Args:
        features: Input feature map [B, H, W, C].
        boxes: RoI boxes [N, 4] in (x1, y1, x2, y2) format.
        output_size: Output size (height, width).
        spatial_scale: Scale factor for boxes.

    Returns:
        Pooled features [N, out_h, out_w, C].
    """
    _batch_size, feat_h, feat_w, channels = features.shape
    _num_boxes = boxes.shape[0]
    out_h, out_w = output_size

    # Scale boxes
    scaled_boxes = boxes * spatial_scale

    # Simple max pooling implementation
    feat_single = features[0]

    def pool_single_box(box: Float[Array, "4"]) -> Float[Array, "out_h out_w channels"]:
        x1, y1, x2, y2 = box

        # Quantize to integer coordinates
        x1 = jnp.clip(jnp.floor(x1).astype(jnp.int32), 0, feat_w - 1)
        y1 = jnp.clip(jnp.floor(y1).astype(jnp.int32), 0, feat_h - 1)
        x2 = jnp.clip(jnp.ceil(x2).astype(jnp.int32), 0, feat_w)
        y2 = jnp.clip(jnp.ceil(y2).astype(jnp.int32), 0, feat_h)

        roi_h = jnp.maximum(y2 - y1, 1)
        roi_w = jnp.maximum(x2 - x1, 1)

        bin_h = roi_h / out_h
        bin_w = roi_w / out_w

        output = jnp.zeros((out_h, out_w, channels))

        for iy in range(out_h):
            for ix in range(out_w):
                # Bin boundaries
                start_y = y1 + jnp.floor(iy * bin_h).astype(jnp.int32)
                start_x = x1 + jnp.floor(ix * bin_w).astype(jnp.int32)
                end_y = y1 + jnp.ceil((iy + 1) * bin_h).astype(jnp.int32)
                end_x = x1 + jnp.ceil((ix + 1) * bin_w).astype(jnp.int32)

                start_y = jnp.clip(start_y, 0, feat_h)
                start_x = jnp.clip(start_x, 0, feat_w)
                end_y = jnp.clip(end_y, 0, feat_h)
                end_x = jnp.clip(end_x, 0, feat_w)

                # Max pool over bin using dynamic slice
                bin_h = end_y - start_y
                bin_w = end_x - start_x
                if bin_h > 0 and bin_w > 0:
                    bin_feat = jax.lax.dynamic_slice(feat_single, (start_y, start_x, 0), (bin_h, bin_w, channels))
                    pooled = jnp.max(bin_feat, axis=(0, 1))
                else:
                    pooled = jnp.zeros(channels)

                output = output.at[iy, ix].set(pooled)

        return output

    pooled_features = jax.vmap(pool_single_box)(scaled_boxes)

    return pooled_features
