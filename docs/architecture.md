# Detectax Architecture

This document provides a detailed overview of the Mask R-CNN architecture as implemented in Detectax.

## Overview

Mask R-CNN is a two-stage object detection and instance segmentation model that extends Faster R-CNN by adding a mask prediction branch.

```
Input Image
    ↓
Backbone (ResNet/ViT)
    ↓
Feature Pyramid Network (FPN)
    ↓
Region Proposal Network (RPN)
    ↓
RoI Align
    ↓
┌─────────────┬──────────────┐
│ Detection   │ Mask         │
│ Head        │ Head         │
└─────────────┴──────────────┘
    ↓              ↓
Bounding Boxes  Instance Masks
```

## Components

### 1. Backbone

The backbone extracts hierarchical features from the input image.

**Supported backbones:**
- ResNet-50 (default)
- ResNet-101
- Vision Transformer (ViT) - planned

**Implementation**: `detectax/models/backbones/`

The backbone produces multi-scale feature maps at different resolutions:
- C2: 1/4 of input size
- C3: 1/8 of input size
- C4: 1/16 of input size
- C5: 1/32 of input size

### 2. Feature Pyramid Network (FPN)

FPN builds a multi-scale feature pyramid by combining low-resolution, semantically strong features with high-resolution, semantically weak features.

**Key operations:**
1. **Top-down pathway**: Upsamples higher-level features
2. **Lateral connections**: Merges with backbone features via 1×1 convolutions
3. **Output**: Multi-scale feature maps {P2, P3, P4, P5, P6}

**Implementation**: `detectax/models/necks/fpn.py`

```python
# FPN output levels
P2: 1/4 resolution  (stride 4)
P3: 1/8 resolution  (stride 8)
P4: 1/16 resolution (stride 16)
P5: 1/32 resolution (stride 32)
P6: 1/64 resolution (stride 64) - max pooling of P5
```

### 3. Region Proposal Network (RPN)

RPN generates object proposals (potential bounding boxes) from FPN features.

**Architecture:**
- Shared 3×3 conv layer
- Classification head (objectness score)
- Regression head (box deltas)

**Anchors:**
- 3 scales: {32², 64², 128², 256², 512²}
- 3 aspect ratios: {1:2, 1:1, 2:1}
- Total: 15 anchors per FPN level

**Training:**
- Positive: IoU > 0.7 with ground truth
- Negative: IoU < 0.3 with ground truth
- Loss: Binary cross-entropy + smooth L1

**Implementation**: `detectax/models/heads/rpn.py`

### 4. RoI Align

RoI Align extracts fixed-size features for each region proposal without quantization errors (unlike RoI Pooling).

**Key features:**
- Bilinear interpolation for precise feature extraction
- Output: 7×7 feature grid for each proposal
- Preserves spatial alignment for mask prediction

**Implementation**: `detectax/models/detectors/roi_align.py`

### 5. Detection Head

The detection head refines proposals and predicts final classes and bounding boxes.

**Architecture:**
- Two fully-connected layers (1024 units each)
- Classification branch: (num_classes + 1) outputs
- Regression branch: 4 × num_classes outputs (class-specific boxes)

**Training:**
- Positive: IoU > 0.5 with ground truth
- Negative: IoU in [0.1, 0.5]
- Loss: Cross-entropy + smooth L1

**Implementation**: `detectax/models/heads/bbox_head.py`

### 6. Mask Head

The mask head predicts instance segmentation masks for each detected object.

**Architecture:**
- Four 3×3 conv layers (256 channels)
- One 2× bilinear upsampling
- 1×1 conv for class-specific mask prediction
- Output: 28×28 mask per instance per class

**Training:**
- Only applied to positive RoIs
- Binary cross-entropy loss (pixel-wise)
- Class-specific masks (no competition between classes)

**Implementation**: `detectax/models/heads/mask_head.py`

## Data Flow

### Training

```
1. Image + Annotations
2. Backbone → FPN features
3. RPN:
   - Generate proposals
   - Compute RPN loss (objectness + box regression)
4. RoI Align: Extract features for proposals
5. Detection Head:
   - Refine proposals
   - Compute detection loss (classification + box regression)
6. Mask Head:
   - Predict masks for positive RoIs
   - Compute mask loss (binary cross-entropy)
7. Total Loss = λ_rpn × L_rpn + λ_det × L_det + λ_mask × L_mask
```

### Inference

```
1. Image
2. Backbone → FPN features
3. RPN → Proposals
4. RoI Align → Proposal features
5. Detection Head → Refined boxes + class scores
6. NMS (Non-Maximum Suppression)
7. Mask Head → Masks for top-k detections
8. Output: Boxes + Labels + Scores + Masks
```

## Loss Functions

### RPN Loss

```
L_rpn = (1/N_cls) Σ L_cls(p_i, p_i*) + λ (1/N_reg) Σ p_i* L_reg(t_i, t_i*)
```

- `L_cls`: Binary cross-entropy (object vs background)
- `L_reg`: Smooth L1 loss for box regression
- `p_i*`: Ground truth label (1 = object, 0 = background)
- `t_i*`: Ground truth box coordinates

### Detection Loss

```
L_det = L_cls(p, u) + λ [u >= 1] L_reg(t^u, v)
```

- `L_cls`: Multi-class cross-entropy
- `L_reg`: Smooth L1 loss
- `u`: Ground truth class
- `v`: Ground truth box

### Mask Loss

```
L_mask = -(1/m²) Σ [y_ij log(ŷ_ij^k) + (1 - y_ij) log(1 - ŷ_ij^k)]
```

- Binary cross-entropy per pixel
- Applied only to ground-truth class `k`
- `m × m`: Mask resolution (28×28)

## Hyperparameters

### Default Configuration

```yaml
model:
  backbone: resnet50
  fpn_channels: 256

  rpn:
    anchor_scales: [32, 64, 128, 256, 512]
    anchor_ratios: [0.5, 1.0, 2.0]
    nms_threshold: 0.7
    train_pre_nms_topk: 2000
    train_post_nms_topk: 1000
    test_pre_nms_topk: 1000
    test_post_nms_topk: 1000

  roi_head:
    num_classes: 80
    roi_size: 7
    roi_sampling_ratio: 2
    bbox_reg_weights: [10.0, 10.0, 5.0, 5.0]

  mask_head:
    num_conv: 4
    conv_channels: 256
    mask_size: 28
```

## Implementation Notes

### JAX/Flax Specifics

- **Functional design**: All modules are pure functions
- **Explicit state**: Parameters passed explicitly, no hidden state
- **PRNG keys**: Random operations require explicit PRNGKey
- **JIT compilation**: Most operations are JIT-compiled for performance
- **Vectorization**: Batch operations use `jax.vmap`

### Reference Implementation

The architecture closely follows the PyTorch reference implementation in `reference/visdet_models`, ensuring mathematical correctness while adapting to JAX idioms.

## Performance Considerations

- **FPN**: Multi-scale features improve detection of small and large objects
- **RoI Align**: Eliminates quantization errors, crucial for mask accuracy
- **Anchor-free alternatives**: Future work may explore FCOS, CenterNet
- **Backbone choices**: Larger backbones (ResNet-101) improve accuracy at cost of speed

## References

- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [Scenic Documentation](https://github.com/google-research/scenic)
