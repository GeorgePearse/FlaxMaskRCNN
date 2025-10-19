#!/bin/bash
# Create remaining Sprint 6-7 tasks (Integration, Training, Data, Evaluation)

cd /home/georgepearse/FlaxMaskRCNN/tasks
mkdir -p sprint_{6,7}

# Sprint 6: Complete Detector Integration
cat > sprint_6/task_6_1_mask_rcnn_detector.json << 'JSON'
{
  "prompt": "Implement Task 6.1: Complete Mask R-CNN Detector Module.\n\nCreate detectax/models/detectors/mask_rcnn.py integrating all components.\n\nRequirements:\n- Flax nn.Module class MaskRCNN\n- Integrate: Backbone → FPN → RPN → Detection Head → Mask Head\n- Forward pass: images → detections + masks\n- Training mode: return losses (rpn_loss + detection_loss + mask_loss)\n- Inference mode: return final predictions\n- Support both modes via training=True/False parameter\n- Full jaxtyping annotations\n\nTests in tests/test_mask_rcnn.py:\n- Full forward pass\n- Training mode loss computation\n- Inference mode predictions\n- Multi-image batching\n\nReferences: reference/visdet_models/, Mask R-CNN paper",
  "cli_name": "codex",
  "files": [
    "/home/georgepearse/FlaxMaskRCNN/ARCHITECTURE_TASKS.md",
    "/home/georgepearse/FlaxMaskRCNN/detectax/models/necks/fpn.py"
  ]
}
JSON

# Sprint 7: Training Infrastructure
cat > sprint_7/task_7_1_coco_loader.json << 'JSON'
{
  "prompt": "Implement Task 7.1: COCO Dataset Loader.\n\nCreate detectax/data/coco_dataset.py for loading COCO format data.\n\nRequirements:\n- Load COCO JSON annotations\n- Parse images, boxes, labels, masks\n- Convert to JAX arrays\n- Support train/val splits\n- Efficient loading with tf.data\n- Validate annotation format\n\nTests: Load CMR COCO dataset at /home/georgepearse/data/cmr/annotations/\n\nReferences: reference/visdet_datasets/coco.py",
  "cli_name": "codex",
  "files": [
    "/home/georgepearse/FlaxMaskRCNN/ARCHITECTURE_TASKS.md",
    "/home/georgepearse/FlaxMaskRCNN/reference/visdet_datasets/coco.py"
  ]
}
JSON

cat > sprint_7/task_7_2_augmentation.json << 'JSON'
{
  "prompt": "Implement Task 7.2: Data Augmentation Pipeline.\n\nCreate detectax/data/augmentations.py for training augmentations.\n\nRequirements:\n- Random horizontal flip (boxes + masks)\n- Random scale jitter (0.8-1.2)\n- Color jitter\n- Normalization (ImageNet stats)\n- Maintain box/mask consistency\n- JAX-compatible or TF ops\n\nTests: Verify augmentation correctness\n\nReferences: reference/visdet_datasets/transforms/",
  "cli_name": "codex",
  "files": [
    "/home/georgepearse/FlaxMaskRCNN/ARCHITECTURE_TASKS.md",
    "/home/georgepearse/FlaxMaskRCNN/reference/visdet_datasets/transforms/transforms.py"
  ]
}
JSON

cat > sprint_7/task_8_1_training_loop.json << 'JSON'
{
  "prompt": "Implement Task 8.1: Training Loop.\n\nCreate detectax/training/train.py for training Mask R-CNN.\n\nRequirements:\n- JAX training loop with jax.jit\n- Optimizer: AdamW with gradient clipping\n- Multi-GPU support with pmap\n- Checkpointing with orbax\n- Logging with tqdm\n- Config-driven (ml_collections)\n\nTests: Smoke test training for 1 iteration\n\nReferences: JAX training examples",
  "cli_name": "codex",
  "files": [
    "/home/georgepearse/FlaxMaskRCNN/ARCHITECTURE_TASKS.md"
  ]
}
JSON

cat > sprint_7/task_8_2_lr_schedule.json << 'JSON'
{
  "prompt": "Implement Task 8.2: Learning Rate Schedule.\n\nCreate detectax/training/schedules.py for LR scheduling.\n\nRequirements:\n- Warmup schedule (linear, 500 steps)\n- Step decay (0.1 at epochs 8, 11 for 12-epoch schedule)\n- Cosine annealing option\n- Config-driven\n\nTests: Verify schedule values\n\nReferences: Optax schedules",
  "cli_name": "codex",
  "files": [
    "/home/georgepearse/FlaxMaskRCNN/ARCHITECTURE_TASKS.md"
  ]
}
JSON

cat > sprint_7/task_9_1_coco_eval.json << 'JSON'
{
  "prompt": "Implement Task 9.1: COCO Evaluator.\n\nCreate detectax/evaluation/coco_evaluator.py for COCO metrics.\n\nRequirements:\n- Wrapper around pycocotools COCOeval\n- Compute bbox AP, mask AP\n- Support IoU thresholds 0.5:0.95\n- Report AP, AP50, AP75, APs, APm, APl\n\nTests: Evaluate dummy predictions\n\nReferences: pycocotools documentation",
  "cli_name": "codex",
  "files": [
    "/home/georgepearse/FlaxMaskRCNN/ARCHITECTURE_TASKS.md"
  ]
}
JSON

echo "Created 6 Sprint 6-7 tasks"
ls -1 sprint_{6,7}/*.json | wc -l
