# Mask R-CNN Architecture Implementation Tasks

**Status**: Task breakdown for Codex implementation via clink
**Created**: 2025-10-19
**Last Updated**: 2025-10-19

## Overview

This document provides a comprehensive, ultra-granular breakdown of all remaining architecture components needed to complete the Mask R-CNN implementation in JAX/Flax. Each task is designed to be independently implementable by Codex via clink.

## Task Categories

1. **Backbone Integration** - Connect existing backbones to FPN
2. **Region Proposal Network (RPN)** - Generate object proposals
3. **Detection Head** - Box classification and regression
4. **Mask Head** - Instance segmentation masks
5. **Complete Detector** - Full Mask R-CNN pipeline
6. **Loss Functions** - Training objectives
7. **Training Infrastructure** - Loops, optimizers, schedulers
8. **Data Pipeline** - COCO dataset loading and augmentation
9. **Evaluation** - COCO metrics computation

---

## Phase 1: Core Detection Components

###  Task 1.1: Anchor Generator
**Priority**: P0 (Blocker for RPN)
**Complexity**: Low
**Dependencies**: None
**Estimated Lines**: ~150

**Description**:
Create `detectrax/models/utils/anchor_generator.py` that generates anchor boxes at multiple scales and aspect ratios for each feature pyramid level.

**Requirements**:
- Generate anchors for FPN levels P2-P6
- Support configurable scales (e.g., [32, 64, 128, 256, 512])
- Support configurable aspect ratios (e.g., [0.5, 1.0, 2.0])
- Output format: (N, 4) where boxes are (x1, y1, x2, y2)
- Pure functional implementation with jax.vmap
- Full type annotations with jaxtyping

**Reference**:
- `reference/visdet_models/dense_heads/anchor_head.py` - PyTorch anchor generation logic
- Mask R-CNN paper Section 3.1

**Tests**:
- Verify anchor counts per level
- Check anchor shapes and scales
- Validate aspect ratios
- Test with different feature map sizes

---

### Task 1.2: Box Encoding/Decoding
**Priority**: P0 (Blocker for RPN and Detection Head)
**Complexity**: Low
**Dependencies**: None
**Estimated Lines**: ~100

**Description**:
Create `detectrax/models/utils/box_coder.py` with functions to encode/decode bounding boxes using the standard Faster R-CNN parameterization.

**Requirements**:
- `encode_boxes(boxes, anchors)`: Convert (x1,y1,x2,y2) to (dx, dy, dw, dh) deltas
- `decode_boxes(deltas, anchors)`: Convert deltas back to (x1,y1,x2,y2)
- Support batched operations
- Configurable weights for delta normalization
- Clip to prevent extreme values
- Full type annotations

**Reference**:
- Faster R-CNN paper Section 3.1.2
- `reference/visdet_models/task_modules/coders/` - PyTorch box coding

**Tests**:
- Round-trip encode/decode identity
- Gradient flow through operations
- Batch processing
- Edge cases (zero-size boxes, clipping)

---

### Task 1.3: NMS (Non-Maximum Suppression)
**Priority**: P0 (Blocker for RPN and Detection Head)
**Complexity**: Medium
**Dependencies**: None
**Estimated Lines**: ~80

**Description**:
Create `detectrax/models/utils/nms.py` implementing batched NMS for post-processing detections.

**Requirements**:
- `nms(boxes, scores, iou_threshold)`: Standard NMS algorithm
- Support batched inputs (different images)
- Use JAX-compatible implementation (no Python loops in main logic)
- Configurable IoU threshold (default 0.7 for RPN, 0.5 for detection)
- Return indices of kept boxes
- Full type annotations

**Reference**:
- `reference/visdet_models/layers/bbox_nms.py` - PyTorch NMS
- Consider using jax.ops or implementing with jax.lax.scan

**Tests**:
- Overlapping boxes are suppressed
- Non-overlapping boxes are kept
- Score ordering is respected
- Batch dimension handling

---

### Task 1.4: IoU Computation
**Priority**: P0 (Blocker for matching and NMS)
**Complexity**: Low
**Dependencies**: None
**Estimated Lines**: ~60

**Description**:
Create `detectrax/models/utils/iou.py` for computing Intersection over Union between box sets.

**Requirements**:
- `box_iou(boxes1, boxes2)`: Pairwise IoU between two sets of boxes
- Vectorized implementation with jax.vmap
- Support (N, 4) and (M, 4) inputs, output (N, M)
- Handle edge cases (zero-area boxes)
- Optional GIoU variant for future use
- Full type annotations

**Reference**:
- Standard IoU formula: intersection / union
- `reference/visdet_models/task_modules/` for reference

**Tests**:
- Perfect overlap (IoU = 1.0)
- No overlap (IoU = 0.0)
- Partial overlap
- Vectorization correctness
- Edge cases

---

## Phase 2: Region Proposal Network (RPN)

### Task 2.1: RPN Head Module
**Priority**: P0 (Critical path)
**Complexity**: Medium
**Dependencies**: Anchor Generator, Box Coder
**Estimated Lines**: ~200

**Description**:
Create `detectrax/models/heads/rpn_head.py` implementing the RPN that predicts objectness and box deltas.

**Requirements**:
- Flax nn.Module with `__call__(features: dict) -> (objectness, deltas)`
- Input: FPN features dict {P2, P3, P4, P5, P6}
- Two parallel 1x1 conv branches:
  - Objectness classification: num_anchors * 1 channels
  - Box regression: num_anchors * 4 channels
- Shared conv before branches (3x3, configurable channels)
- Initialize biases for objectness (see Focal Loss paper)
- Support anchor_scales and anchor_ratios configuration
- Full type annotations

**Reference**:
- `reference/visdet_models/dense_heads/anchor_head.py`
- Faster R-CNN paper Section 3.1

**Tests**:
- Output shapes for each FPN level
- Gradient flow
- Anchor alignment
- Integration with anchor generator

---

### Task 2.2: RPN Target Assignment
**Priority**: P0 (Blocker for RPN training)
**Complexity**: High
**Dependencies**: IoU Computation, Anchor Generator
**Estimated Lines**: ~250

**Description**:
Create `detectrax/models/task_modules/assigners/rpn_assigner.py` that assigns ground truth boxes to anchors for RPN training.

**Requirements**:
- Assign positive labels to anchors with IoU > 0.7 with any GT box
- Assign negative labels to anchors with IoU < 0.3 with all GT boxes
- Ignore anchors with 0.3 ≤ IoU ≤ 0.7
- Handle crowded/ignore regions
- Support batched operations
- Return assigned labels and target boxes
- Pure functional implementation
- Full type annotations

**Reference**:
- `reference/visdet_models/task_modules/assigners/max_iou_assigner.py`
- Faster R-CNN paper Section 3.1.2

**Tests**:
- High IoU anchors are positive
- Low IoU anchors are negative
- Intermediate IoU anchors are ignored
- Batch processing
- Empty GT case

---

### Task 2.3: RPN Proposal Generation
**Priority**: P0 (Blocker for detection heads)
**Complexity**: Medium
**Dependencies**: Box Decoder, NMS
**Estimated Lines**: ~180

**Description**:
Create `detectrax/models/task_modules/proposal_generator.py` that converts RPN outputs to final proposals.

**Requirements**:
- Decode box deltas to actual boxes
- Apply NMS with IoU threshold 0.7
- Select top-k proposals (1000 training, 1000 inference)
- Clip boxes to image boundaries
- Filter very small boxes (min_size parameter)
- Return (num_proposals, 5) where each row is (batch_idx, x1, y1, x2, y2)
- Support different behavior for training vs inference
- Full type annotations

**Reference**:
- `reference/visdet_models/dense_heads/anchor_head.py` - `get_proposals` method
- Faster R-CNN paper

**Tests**:
- NMS reduces overlapping proposals
- Top-k selection works
- Box clipping
- Small box filtering
- Training vs inference modes

---

## Phase 3: Detection Head (Fast R-CNN)

### Task 3.1: RoI Head Base Class
**Priority**: P1
**Complexity**: Low
**Dependencies**: RoI Align (completed)
**Estimated Lines**: ~100

**Description**:
Create `detectrax/models/roi_heads/base_roi_head.py` with shared logic for detection and mask heads.

**Requirements**:
- Abstract base using Flax nn.Module
- Common RoI extraction using RoI Align
- Feature pooling from FPN levels
- Support for multi-level feature extraction
- Placeholder for subclass implementations
- Full type annotations

**Reference**:
- `reference/visdet_models/roi_heads/base_roi_head.py`
- Mask R-CNN paper Section 3

**Tests**:
- Basic instantiation
- Feature extraction shapes
- FPN level routing

---

### Task 3.2: Box Head (Classification + Regression)
**Priority**: P1 (Critical path)
**Complexity**: Medium
**Dependencies**: RoI Head Base, Box Coder
**Estimated Lines**: ~250

**Description**:
Create `detectrax/models/roi_heads/bbox_head.py` for classifying RoIs and refining boxes.

**Requirements**:
- Flax nn.Module taking pooled RoI features (N, 7, 7, C)
- Two FC layers (configurable hidden dim, default 1024)
- Two output branches:
  - Classification: (N, num_classes + 1) logits (including background)
  - Box regression: (N, num_classes * 4) deltas (class-specific)
- ReLU activations
- Support for class-agnostic regression (optional)
- Full type annotations

**Reference**:
- `reference/visdet_models/roi_heads/bbox_heads/bbox_head.py`
- Fast R-CNN paper

**Tests**:
- Output shapes
- Gradient flow
- Class-specific vs class-agnostic modes
- Integration with RoI features

---

### Task 3.3: Detection Target Assignment
**Priority**: P1 (Blocker for detection training)
**Complexity**: High
**Dependencies**: IoU Computation
**Estimated Lines**: ~300

**Description**:
Create `detectrax/models/task_modules/assigners/detection_assigner.py` for assigning GT to proposals during training.

**Requirements**:
- Assign positive labels to proposals with IoU ≥ 0.5 with any GT
- Assign negative labels to proposals with IoU < 0.5 with all GT
- Sample balanced batches (e.g., 512 proposals, 25% positive)
- Assign target class labels and box targets
- Handle empty GT cases
- Support cascade refinement (optional, for later)
- Pure functional with vmaps
- Full type annotations

**Reference**:
- `reference/visdet_models/task_modules/assigners/max_iou_assigner.py`
- Fast R-CNN paper Section 2.1

**Tests**:
- IoU-based assignment
- Balanced sampling
- Class label assignment
- Box target encoding
- Empty GT handling

---

### Task 3.4: Detection Post-Processing
**Priority**: P1
**Complexity**: Medium
**Dependencies**: NMS, Box Decoder
**Estimated Lines**: ~200

**Description**:
Create `detectrax/models/task_modules/detection_postprocess.py` for converting raw detections to final outputs.

**Requirements**:
- Decode class-specific box deltas
- Apply per-class NMS (IoU threshold 0.5)
- Filter by confidence threshold (e.g., 0.05)
- Select top-k detections per image (e.g., 100)
- Return format: boxes (N, 4), scores (N,), labels (N,), batch_indices (N,)
- Support batched inputs
- Full type annotations

**Reference**:
- `reference/visdet_models/roi_heads/standard_roi_head.py`
- Faster R-CNN inference

**Tests**:
- NMS per class
- Confidence filtering
- Top-k selection
- Batch processing

---

## Phase 4: Mask Head

### Task 4.1: Mask Head Module
**Priority**: P2
**Complexity**: Medium
**Dependencies**: RoI Head Base
**Estimated Lines**: ~200

**Description**:
Create `detectrax/models/roi_heads/mask_head.py` for predicting instance segmentation masks.

**Requirements**:
- Flax nn.Module taking RoI features (N, 14, 14, C)
- Four consecutive 3x3 convs (256 channels each)
- 2x upsampling via transposed conv
- Final 1x1 conv to num_classes channels
- Output: (N, num_classes, 28, 28) mask logits
- Per-class masks (predict only for assigned class during training)
- Full type annotations

**Reference**:
- `reference/visdet_models/roi_heads/mask_heads/fcn_mask_head.py`
- Mask R-CNN paper Section 3.1

**Tests**:
- Output shape (28x28 masks)
- Gradient flow
- Per-class mask extraction
- Integration with detection head

---

### Task 4.2: Mask Target Generation
**Priority**: P2
**Complexity**: High
**Dependencies**: RoI Align
**Estimated Lines**: ~250

**Description**:
Create `detectrax/models/task_modules/mask_targets.py` for generating mask training targets from GT polygons/masks.

**Requirements**:
- Convert COCO polygon annotations to binary masks
- Crop masks to RoI boxes
- Resize to 28x28 via bilinear interpolation
- Handle multiple instances per image
- Support RLE mask format (COCO compressed)
- Return binary masks (N, 28, 28)
- Pure functional where possible
- Full type annotations

**Reference**:
- `reference/visdet_models/task_modules/` mask handling
- pycocotools for mask utilities

**Tests**:
- Polygon to mask conversion
- Cropping and resizing
- RLE decoding
- Multiple instances

---

### Task 4.3: Mask Post-Processing
**Priority**: P2
**Complexity**: Medium
**Dependencies**: None
**Estimated Lines**: ~150

**Description**:
Create `detectrax/models/task_modules/mask_postprocess.py` for converting mask logits to final instance masks.

**Requirements**:
- Threshold mask logits at 0.5
- Resize from 28x28 to original RoI size
- Paste masks into full image canvas
- Handle overlapping instances (higher score wins)
- Return binary masks per instance
- Support batched processing
- Full type annotations

**Reference**:
- Mask R-CNN paper Section 3.1
- `reference/visdet_models/` mask utilities

**Tests**:
- Thresholding
- Resizing quality
- Pasting into image
- Overlap handling

---

## Phase 5: Loss Functions

### Task 5.1: RPN Loss
**Priority**: P0 (Blocker for RPN training)
**Complexity**: Medium
**Dependencies**: Box Coder
**Estimated Lines**: ~120

**Description**:
Create `detectrax/models/losses/rpn_loss.py` implementing RPN training loss.

**Requirements**:
- Binary cross-entropy for objectness classification
- Smooth L1 loss for box regression (only on positive anchors)
- Ignore loss for neutral anchors
- Normalize by number of anchors
- Return dict with separate classification and regression losses
- Full type annotations

**Reference**:
- `reference/visdet_models/losses/cross_entropy_loss.py`
- `reference/visdet_models/losses/smooth_l1_loss.py`
- Faster R-CNN paper

**Tests**:
- Classification loss magnitude
- Regression loss only on positives
- Ignored anchors don't contribute
- Gradient flow

---

### Task 5.2: Detection Loss
**Priority**: P1 (Blocker for detection training)
**Complexity**: Medium
**Dependencies**: Box Coder
**Estimated Lines**: ~150

**Description**:
Create `detectrax/models/losses/detection_loss.py` for Fast R-CNN detection loss.

**Requirements**:
- Cross-entropy loss for classification (num_classes + 1 with background)
- Smooth L1 loss for box regression (class-specific, only on positives)
- Ignore background class in regression
- Normalize losses appropriately
- Return dict with separate components
- Support label smoothing (optional)
- Full type annotations

**Reference**:
- `reference/visdet_models/losses/`
- Fast R-CNN paper Section 2.2

**Tests**:
- Classification loss shape
- Regression only on positives
- Class-specific regression
- Loss magnitudes

---

### Task 5.3: Mask Loss
**Priority**: P2 (Blocker for mask training)
**Complexity**: Low
**Dependencies**: None
**Estimated Lines**: ~80

**Description**:
Create `detectrax/models/losses/mask_loss.py` for mask prediction loss.

**Requirements**:
- Binary cross-entropy per pixel
- Only compute loss for assigned class mask
- Average over positive RoIs
- Ignore negative proposals
- Return scalar loss
- Full type annotations

**Reference**:
- `reference/visdet_models/roi_heads/mask_heads/fcn_mask_head.py` - loss method
- Mask R-CNN paper

**Tests**:
- Per-pixel BCE
- Class-specific masking
- Positive-only training
- Loss magnitude

---

## Phase 6: Complete Mask R-CNN Detector

### Task 6.1: Mask R-CNN Detector Module
**Priority**: P1 (Integration task)
**Complexity**: High
**Dependencies**: All previous components
**Estimated Lines**: ~400

**Description**:
Create `detectrax/models/detectors/mask_rcnn.py` integrating all components into full detector.

**Requirements**:
- Flax nn.Module combining backbone + FPN + RPN + detection head + mask head
- Forward pass for training: return losses
- Forward pass for inference: return final detections and masks
- Handle image preprocessing (normalization, resizing)
- Multi-scale testing support (optional)
- Configuration via ml_collections.ConfigDict
- Clean API: `MaskRCNN(config)`
- Full type annotations

**Architecture Flow**:
```
Input Image (H, W, 3)
  ↓
Backbone (ResNet-50/101)
  ↓
FPN (P2, P3, P4, P5, P6)
  ↓
RPN → Proposals (N, 5)
  ↓
RoI Align (7x7 for detection, 14x14 for masks)
  ↓
Detection Head → Boxes (M, 4), Classes (M,)
  ↓
Mask Head → Masks (M, 28, 28)
```

**Reference**:
- `reference/visdet_models/` for integration patterns
- Mask R-CNN paper for full architecture

**Tests**:
- Full forward pass (training mode)
- Full forward pass (inference mode)
- Loss computation
- End-to-end gradient flow
- Different backbone configurations

---

## Phase 7: Data Pipeline

### Task 7.1: COCO Dataset Loader
**Priority**: P1 (Blocker for training)
**Complexity**: High
**Dependencies**: None (uses TensorFlow)
**Estimated Lines**: ~350

**Description**:
Create `detectrax/data/coco_dataset.py` for loading and parsing COCO format datasets.

**Requirements**:
- Load from COCO JSON annotations
- Parse images, boxes, classes, masks, keypoints
- Support train/val splits
- Handle COCO RLE mask format
- Convert annotations to detector format
- Iterator interface compatible with JAX training loops
- Use tf.data for efficiency (prefetch, parallel map)
- Full type annotations

**Reference**:
- `reference/visdet_datasets/coco.py` - PyTorch COCO loader
- tensorflow_datasets COCO implementation
- pycocotools for parsing

**Tests**:
- Load sample annotations
- Parse boxes correctly
- Decode masks
- Batch generation
- Shuffling and repeating

---

### Task 7.2: Data Augmentation Pipeline
**Priority**: P1 (Important for training)
**Complexity**: Medium
**Dependencies**: COCO Dataset Loader
**Estimated Lines**: ~300

**Description**:
Create `detectrax/data/augmentation.py` with training data augmentations.

**Requirements**:
- Random horizontal flip (with box + mask flipping)
- Random scale jittering (0.8x to 1.2x)
- Color jittering (brightness, contrast, saturation)
- Random crop (optional)
- Normalization (ImageNet stats)
- Resize to fixed size (e.g., 800x1333 or 1024x1024)
- Maintain box and mask consistency
- Use TensorFlow ops for GPU efficiency
- Full type annotations

**Reference**:
- `reference/visdet_datasets/transforms/` - PyTorch augmentations
- Detectron2 augmentation pipeline

**Tests**:
- Flip preserves instance alignment
- Scaling maintains aspect ratios
- Box coordinates remain valid
- Masks match boxes after transforms

---

## Phase 8: Training Infrastructure

### Task 8.1: Training Loop
**Priority**: P1 (Critical for training)
**Complexity**: High
**Dependencies**: Complete Detector, Data Pipeline
**Estimated Lines**: ~400

**Description**:
Create `detectrax/training/train.py` with full training loop using JAX patterns.

**Requirements**:
- Use jax.pmap for multi-GPU training
- Optax optimizer setup (SGD with momentum)
- Learning rate schedule (warmup + cosine decay)
- Gradient clipping
- Loss accumulation across devices
- Checkpointing with orbax-checkpoint
- Logging (console + TensorBoard/W&B)
- Metrics tracking (loss components, learning rate)
- Config-driven (ml_collections)
- Resumable training
- Full type annotations

**Reference**:
- Scenic training infrastructure: `scenic/train_lib/`
- JAX official examples: `jax/example_libraries/`

**Tests**:
- Single GPU training
- Multi-GPU training (if available)
- Checkpointing and resuming
- Gradient accumulation
- LR scheduling

---

### Task 8.2: Learning Rate Schedule
**Priority**: P1 (Important for convergence)
**Complexity**: Low
**Dependencies**: None
**Estimated Lines**: ~100

**Description**:
Create `detectrax/training/lr_schedule.py` with learning rate scheduling functions.

**Requirements**:
- Warmup schedule (linear rampup for first N iterations)
- Cosine decay schedule
- Step decay schedule (drop at specified milestones)
- Polynomial decay (optional)
- Compatible with Optax
- Pure functional implementations
- Full type annotations

**Reference**:
- Optax schedules: `optax.warmup_cosine_decay_schedule`
- Detectron2 LR scheduler

**Tests**:
- Warmup behavior
- Decay curves
- Combined warmup + decay
- Edge cases (step 0, very large steps)

---

## Phase 9: Evaluation

### Task 9.1: COCO Evaluator
**Priority**: P2 (Needed for validation)
**Complexity**: Medium
**Dependencies**: Complete Detector
**Estimated Lines**: ~250

**Description**:
Create `detectrax/evaluation/coco_evaluator.py` for computing COCO metrics.

**Requirements**:
- Compute box AP (AP, AP50, AP75, APs, APm, APl)
- Compute mask AP (same metrics)
- Use pycocotools for metric calculation
- Collect predictions over full val set
- Format predictions for COCO evaluator
- Report per-class metrics (optional)
- Full type annotations

**Reference**:
- pycocotools.cocoeval
- Detectron2 COCO evaluator

**Tests**:
- Perfect predictions (AP = 1.0)
- Random predictions (low AP)
- Format compatibility with COCO API
- Batch accumulation

---

### Task 9.2: Visualization Utilities
**Priority**: P3 (Nice to have)
**Complexity**: Low
**Dependencies**: Complete Detector
**Estimated Lines**: ~200

**Description**:
Create `detectrax/utils/visualization.py` for drawing predictions and GT on images.

**Requirements**:
- Draw bounding boxes with class labels
- Overlay instance masks with transparency
- Color-coded by class or instance
- Save to image files
- Support batched inputs
- Use matplotlib or PIL
- Full type annotations

**Reference**:
- Detectron2 visualizer
- Standard visualization libraries

**Tests**:
- Single image visualization
- Batch visualization
- Output image format

---

## Task Execution Strategy

### Recommended Order for Codex Implementation

**Sprint 1: Foundational Utilities (Week 1)**
1. Task 1.1: Anchor Generator
2. Task 1.2: Box Encoding/Decoding
3. Task 1.3: NMS
4. Task 1.4: IoU Computation

**Sprint 2: RPN Components (Week 1-2)**
5. Task 2.1: RPN Head Module
6. Task 2.2: RPN Target Assignment
7. Task 2.3: RPN Proposal Generation
8. Task 5.1: RPN Loss

**Sprint 3: Detection Head (Week 2-3)**
9. Task 3.1: RoI Head Base Class
10. Task 3.2: Box Head
11. Task 3.3: Detection Target Assignment
12. Task 3.4: Detection Post-Processing
13. Task 5.2: Detection Loss

**Sprint 4: Mask Head (Week 3)**
14. Task 4.1: Mask Head Module
15. Task 4.2: Mask Target Generation
16. Task 4.3: Mask Post-Processing
17. Task 5.3: Mask Loss

**Sprint 5: Integration (Week 4)**
18. Task 6.1: Mask R-CNN Detector Module

**Sprint 6: Data + Training (Week 4-5)**
19. Task 7.1: COCO Dataset Loader
20. Task 7.2: Data Augmentation Pipeline
21. Task 8.1: Training Loop
22. Task 8.2: Learning Rate Schedule

**Sprint 7: Evaluation (Week 5)**
23. Task 9.1: COCO Evaluator
24. Task 9.2: Visualization Utilities

### Codex Invocation Template

For each task, use this template with clink:

```python
mcp__zen__clink(
    prompt=f"""
    Implement {TASK_NAME} as described in ARCHITECTURE_TASKS.md.

    Requirements:
    {PASTE_REQUIREMENTS_FROM_TASK}

    Technical constraints:
    - Use JAX/Flax patterns (pure functions, explicit PRNG keys)
    - Full jaxtyping annotations: Float[Array, "batch height width channels"]
    - Follow existing FPN implementation style
    - Include comprehensive docstrings
    - Handle all edge cases mentioned

    Create:
    1. Implementation file at specified path
    2. Comprehensive unit tests in tests/
    3. Update __init__.py exports if needed

    Reference implementation patterns from:
    {REFERENCE_FILES}
    """,
    cli_name="codex",
    files=[
        # Absolute paths to reference files
        # Absolute paths to related implementations
    ]
)
```

### Validation After Each Task

After Codex completes each task:
1. Run tests: `uv run pytest tests/test_XXX.py -v`
2. Check types: `uv run pyright detectrax/`
3. Run linting: `uv run ruff check detectrax/`
4. Review implementation for JAX best practices
5. Commit with descriptive message
6. Update this checklist

---

## Progress Tracking

- [ ] Sprint 1: Foundational Utilities (0/4 tasks)
- [ ] Sprint 2: RPN Components (0/4 tasks)
- [ ] Sprint 3: Detection Head (0/5 tasks)
- [ ] Sprint 4: Mask Head (0/4 tasks)
- [ ] Sprint 5: Integration (0/1 tasks)
- [ ] Sprint 6: Data + Training (0/4 tasks)
- [ ] Sprint 7: Evaluation (0/2 tasks)

**Total Progress**: 0/24 tasks (0%)

---

## Notes

- Each task is designed to be completed in 1-4 hours by Codex
- Dependencies are clearly marked - respect the order
- All implementations must pass tests before proceeding
- Reference PyTorch visdet code for mathematical correctness
- Implement idiomatically in JAX/Flax, don't just port PyTorch code
- Keep context fresh by using separate clink calls per task
- Use continuation_id sparingly - prefer fresh context for unrelated tasks

## Success Criteria

The architecture is complete when:
1. All 24 tasks implemented and tested
2. Full Mask R-CNN forward pass works end-to-end
3. Training loop can run for multiple iterations without errors
4. Evaluation produces COCO metrics
5. All tests passing
6. Type checking passes (pyright strict mode)
7. Pre-commit hooks pass

**Target**: Reach 50% task completion (12/24) within 2 weeks using Codex.
