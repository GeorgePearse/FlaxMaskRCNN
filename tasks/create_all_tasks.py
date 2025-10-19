#!/usr/bin/env python3
"""Generate all remaining task JSON files for Sprints 3-7."""

import json
from pathlib import Path

TASKS_DIR = Path("/home/georgepearse/FlaxMaskRCNN/tasks")
REPO_ROOT = Path("/home/georgepearse/FlaxMaskRCNN")

# Sprint 3: Detection Head (RoI Head)
SPRINT_3_TASKS = [
    {
        "file": "sprint_3/task_3_1_roi_head_base.json",
        "task_name": "Task 3.1: RoI Head Base Class",
        "output_file": "detectax/models/roi_heads/base_roi_head.py",
        "test_file": "tests/test_roi_head.py",
        "description": """Implement RoI Head base class for two-stage detectors.

Requirements:
- Flax nn.Module class BaseRoIHead
- Abstract methods for box_head, get_targets, get_predictions, loss
- RoI extraction using RoI Align from detectax.models.layers.roi_align
- Support multi-scale FPN features (P2-P6)
- Configurable RoI resolution (7x7, 14x14)
- Full jaxtyping annotations
- Virtual methods pattern (subclasses override)

Tests:
- RoI extraction from FPN features
- Multi-scale handling
- Batch processing
- Abstract method enforcement""",
        "references": ["reference/visdet_models/roi_heads/base_roi_head.py", "detectax/models/layers/roi_align.py"],
    },
    {
        "file": "sprint_3/task_3_2_box_head.json",
        "task_name": "Task 3.2: Box Head (Classification + Regression)",
        "output_file": "detectax/models/roi_heads/bbox_heads/bbox_head.py",
        "test_file": "tests/test_bbox_head.py",
        "description": """Implement Box Head for final classification and box regression.

Requirements:
- Flax nn.Module class BBoxHead extending BaseRoIHead
- Two FC layers (1024 hidden dim) after RoI features (7x7x256 → flatten → FC)
- Two output branches:
  - Classification: FC → num_classes logits
  - Regression: FC → num_classes * 4 box deltas (class-specific)
- Initialize regression head with small std (0.001)
- Support class-agnostic or class-specific regression
- Full jaxtyping annotations

Tests:
- Output shapes (batch, num_rois, num_classes) and (batch, num_rois, num_classes*4)
- Gradient flow
- Class-specific vs class-agnostic modes
- Integration with RoI extraction""",
        "references": [
            "reference/visdet_models/roi_heads/bbox_heads/bbox_head.py",
            "reference/visdet_models/roi_heads/bbox_heads/convfc_bbox_head.py",
        ],
    },
    {
        "file": "sprint_3/task_3_3_detection_assigner.json",
        "task_name": "Task 3.3: Detection Target Assignment",
        "output_file": "detectax/models/task_modules/assigners/detection_assigner.py",
        "test_file": "tests/test_detection_assigner.py",
        "description": """Implement target assignment for detection head (2nd stage).

Requirements:
- Function assign_detection_targets(proposals, gt_boxes, gt_labels, pos_iou_threshold=0.5, neg_iou_threshold=0.5)
- Compute IoU between proposals and GT boxes
- Assign positive (label=gt_class) if IoU >= 0.5
- Assign negative (label=0 background) if IoU < 0.5
- For positives, compute target box deltas
- Handle class-specific regression (deltas per class)
- Support batched ops with vmap
- Return: (labels, target_deltas, weights)

Tests:
- Positive/negative assignment
- Background class handling
- Target delta computation
- Class-specific regression targets""",
        "references": ["reference/visdet_models/task_modules/assigners/max_iou_assigner.py"],
    },
    {
        "file": "sprint_3/task_3_4_detection_postprocess.json",
        "task_name": "Task 3.4: Detection Post-Processing",
        "output_file": "detectax/models/task_modules/post_processors/detection_postprocessor.py",
        "test_file": "tests/test_detection_postprocessor.py",
        "description": """Implement post-processing for final detections.

Requirements:
- Function postprocess_detections(proposals, cls_scores, box_deltas, image_shape, score_threshold=0.05, nms_threshold=0.5, max_per_image=100)
- Apply softmax to cls_scores
- Decode box deltas to final boxes
- Filter by score_threshold
- Apply per-class NMS
- Take top max_per_image detections
- Return: (boxes, scores, labels)

Tests:
- Score thresholding
- Per-class NMS
- Top-k selection
- Multi-image batching""",
        "references": ["reference/visdet_models/test_time_augs/merge_augs.py"],
    },
]

# Sprint 4: Mask Head
SPRINT_4_TASKS = [
    {
        "file": "sprint_4/task_4_1_mask_head.json",
        "task_name": "Task 4.1: Mask Head Module",
        "output_file": "detectax/models/roi_heads/mask_heads/fcn_mask_head.py",
        "test_file": "tests/test_mask_head.py",
        "description": """Implement FCN Mask Head for instance segmentation.

Requirements:
- Flax nn.Module class FCNMaskHead
- Input: RoI features (14x14x256) from RoI Align
- Architecture: 4x (Conv 3x3, 256 channels, ReLU) → ConvTranspose 2x2 (stride 2) → Conv 1x1 (num_classes channels)
- Output: (batch, num_rois, 28, 28, num_classes) masks
- Use nn.ConvTranspose for upsampling
- Per-class masks (predict mask for each class separately)
- Full jaxtyping annotations

Tests:
- Output shape (28x28 masks)
- Upsampling correctness
- Per-class mask generation
- Gradient flow""",
        "references": ["reference/visdet_models/roi_heads/mask_heads/fcn_mask_head.py"],
    },
    {
        "file": "sprint_4/task_4_2_mask_targets.json",
        "task_name": "Task 4.2: Mask Target Generation",
        "output_file": "detectax/models/task_modules/mask_target_generator.py",
        "test_file": "tests/test_mask_targets.py",
        "description": """Implement mask target generation for training.

Requirements:
- Function generate_mask_targets(positive_proposals, gt_boxes, gt_masks, mask_size=28)
- Match positive proposals to GT instances
- Crop and resize GT masks to proposal boxes
- Resize to mask_size x mask_size (28x28)
- Handle polygon and RLE mask formats
- Use bilinear interpolation for resizing
- Return: (mask_targets: Float[Array, \"num_positives 28 28\"])

Tests:
- Mask cropping accuracy
- Resizing correctness
- Polygon mask handling
- RLE mask handling""",
        "references": ["reference/visdet_models/roi_heads/mask_heads/"],
    },
    {
        "file": "sprint_4/task_4_3_mask_postprocess.json",
        "task_name": "Task 4.3: Mask Post-Processing",
        "output_file": "detectax/models/task_modules/post_processors/mask_postprocessor.py",
        "test_file": "tests/test_mask_postprocess.py",
        "description": """Implement mask post-processing for inference.

Requirements:
- Function postprocess_masks(mask_logits, detected_boxes, image_shape, threshold=0.5)
- Select masks for detected class only (class-specific)
- Resize masks from 28x28 to box size
- Paste masks into full image
- Apply sigmoid + threshold (0.5)
- Return binary masks at full image resolution

Tests:
- Mask resizing
- Pasting into image
- Thresholding
- Multi-instance handling""",
        "references": ["reference/visdet_models/roi_heads/mask_heads/"],
    },
]

# Sprint 5: Loss Functions
SPRINT_5_TASKS = [
    {
        "file": "sprint_5/task_5_1_rpn_loss.json",
        "task_name": "Task 5.1: RPN Loss",
        "output_file": "detectax/models/losses/rpn_loss.py",
        "test_file": "tests/test_rpn_loss.py",
        "description": """Implement RPN training loss.

Requirements:
- Function rpn_loss(objectness_pred, box_deltas_pred, objectness_targets, box_delta_targets, weights)
- Classification loss: Binary cross-entropy for objectness
- Regression loss: Smooth L1 loss for box deltas (only on positives)
- Balance positive/negative samples (1:1 ratio, sample 256 anchors)
- Weight losses by number of anchors
- Return: (total_loss, cls_loss, reg_loss)

Tests:
- Objectness BCE loss
- Box regression loss
- Sample balancing
- Gradient flow""",
        "references": ["reference/visdet_models/losses/"],
    },
    {
        "file": "sprint_5/task_5_2_detection_loss.json",
        "task_name": "Task 5.2: Detection Loss",
        "output_file": "detectax/models/losses/detection_loss.py",
        "test_file": "tests/test_detection_loss.py",
        "description": """Implement detection head training loss.

Requirements:
- Function detection_loss(cls_scores, box_deltas, cls_targets, box_delta_targets, weights)
- Classification: Cross-entropy loss (num_classes + 1 for background)
- Regression: Smooth L1 loss (only on foreground, class-specific)
- Sample 512 RoIs per image (positive:negative = 1:3)
- Return: (total_loss, cls_loss, reg_loss)

Tests:
- Multi-class CE loss
- Class-specific regression
- Sample balancing
- Background class handling""",
        "references": ["reference/visdet_models/losses/cross_entropy_loss.py", "reference/visdet_models/losses/smooth_l1_loss.py"],
    },
    {
        "file": "sprint_5/task_5_3_mask_loss.json",
        "task_name": "Task 5.3: Mask Loss",
        "output_file": "detectax/models/losses/mask_loss.py",
        "test_file": "tests/test_mask_loss.py",
        "description": """Implement mask head training loss.

Requirements:
- Function mask_loss(mask_pred, mask_targets, positive_indices)
- Binary cross-entropy per-pixel
- Compute loss only on positive RoIs (matched to GT)
- Only supervise mask for ground-truth class
- Average over positive RoIs and pixels
- Return: mask_loss scalar

Tests:
- Per-pixel BCE
- Positive-only supervision
- Class-specific mask loss
- Gradient flow""",
        "references": ["reference/visdet_models/losses/"],
    },
]


def create_task_json(task_spec):
    """Create a task JSON file from specification."""
    task_dir = TASKS_DIR / Path(task_spec["file"]).parent
    task_dir.mkdir(parents=True, exist_ok=True)

    task_file = TASKS_DIR / task_spec["file"]

    prompt = f"""Implement {task_spec["task_name"]} as described in ARCHITECTURE_TASKS.md.

Create {task_spec["output_file"]} implementing the required functionality.

{task_spec["description"]}

Also create comprehensive tests in {task_spec["test_file"]}.

Update relevant __init__.py exports.

Reference implementations: {", ".join(task_spec["references"])}"""

    task_json = {
        "prompt": prompt,
        "cli_name": "codex",
        "files": [str(REPO_ROOT / "ARCHITECTURE_TASKS.md")] + [str(REPO_ROOT / ref) for ref in task_spec["references"]],
    }

    with open(task_file, "w") as f:
        json.dump(task_json, f, indent=2)

    print(f"Created: {task_file}")


# Create all tasks
print("Creating Sprint 3-5 task definitions...")
for task in SPRINT_3_TASKS + SPRINT_4_TASKS + SPRINT_5_TASKS:
    create_task_json(task)

print(f"\n✓ Created {len(SPRINT_3_TASKS + SPRINT_4_TASKS + SPRINT_5_TASKS)} task definitions")
print("Sprint 3 (Detection Head): 4 tasks")
print("Sprint 4 (Mask Head): 3 tasks")
print("Sprint 5 (Losses): 3 tasks")
