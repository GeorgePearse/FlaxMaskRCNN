#!/usr/bin/env python3
"""Flax-based training script for Mask R-CNN.

This script uses the production-ready Flax training infrastructure from
detectrax.training.train, which provides:
- Proper TrainState management with Flax structs
- Orbax checkpointing for model persistence
- Multi-device support via pmap
- AdamW optimizer with gradient clipping
- Progress tracking with tqdm

Usage:
    # CMR dataset with defaults
    python train.py --cmr

    # Custom dataset
    python train.py --annotations /path/to/train.json --images /path/to/images

    # With custom hyperparameters
    python train.py --cmr --batch-size 4 --lr 1e-4 --num-steps 10000
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image

from detectrax.data.coco_utils import get_num_classes_from_coco
from detectrax.models.detectors.mask_rcnn import MaskRCNN, MaskRCNNConfig, MaskRCNNTargets
from detectrax.training.train import train
from ml_collections import ConfigDict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mask R-CNN with Flax infrastructure")

    # Dataset arguments
    parser.add_argument("--cmr", action="store_true", help="Use default CMR dataset paths")
    parser.add_argument("--annotations", type=str, help="Path to COCO annotations JSON file")
    parser.add_argument("--images", type=str, help="Path to images directory")

    # Model arguments
    parser.add_argument("--num-proposals", type=int, default=1000, help="Number of proposals (default: 1000)")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Score threshold (default: 0.05)")
    parser.add_argument(
        "--class-agnostic-bbox", action="store_false", dest="class_agnostic_bbox", help="Disable class-agnostic bbox (use per-class instead)"
    )
    parser.set_defaults(class_agnostic_bbox=True)

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of training steps (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps (default: 100)")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to load (default: all)")

    # Logging and checkpointing
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps (default: 10)")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save checkpoint every N steps (default: 500)")
    parser.add_argument("--max-checkpoints", type=int, default=3, help="Max checkpoints to keep (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # Handle CMR preset
    if args.cmr:
        args.annotations = "/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json"
        args.images = "/home/georgepearse/data/images"

    # Validate required arguments
    if not args.annotations or not args.images:
        parser.error("Either --cmr or both --annotations and --images must be provided")

    return args


def load_coco_batch(annotation_file: str, image_dir: str, batch_size: int = 2, max_images: int | None = None):
    """Load batches from COCO dataset."""
    import json

    with open(annotation_file) as f:
        data = json.load(f)

    images_data = data["images"][:max_images] if max_images else data["images"]
    annotations_data = data["annotations"]

    # Group annotations by image
    img_to_anns = {}
    for ann in annotations_data:
        img_id = ann["image_id"]
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    batches = []
    for i in range(0, len(images_data), batch_size):
        batch_imgs = images_data[i : i + batch_size]

        images_list = []
        boxes_list = []
        labels_list = []
        masks_list = []

        for img_data in batch_imgs:
            # Load and resize image
            img_path = Path(image_dir) / img_data["file_name"]
            if not img_path.exists():
                print(f"Warning: {img_path} not found, skipping")
                continue

            img = Image.open(img_path).convert("RGB")
            img = img.resize((512, 512))
            img_array = np.array(img, dtype=np.float32) / 255.0
            images_list.append(img_array)

            # Get annotations for this image
            img_id = img_data["id"]
            anns = img_to_anns.get(img_id, [])

            if len(anns) == 0:
                # Add dummy annotation
                boxes_list.append(np.array([[100, 100, 200, 200]], dtype=np.float32))
                labels_list.append(np.array([1], dtype=np.int32))
                masks_list.append(np.ones((1, 28, 28), dtype=np.float32))
            else:
                boxes = []
                labels = []
                masks = []

                for ann in anns[:5]:  # Max 5 instances per image
                    bbox = ann["bbox"]  # COCO format: [x, y, w, h]
                    # Convert to [x1, y1, x2, y2] and scale to 512x512
                    x1 = bbox[0] * 512 / img_data["width"]
                    y1 = bbox[1] * 512 / img_data["height"]
                    x2 = (bbox[0] + bbox[2]) * 512 / img_data["width"]
                    y2 = (bbox[1] + bbox[3]) * 512 / img_data["height"]
                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann["category_id"])
                    # Create dummy mask for now
                    masks.append(np.ones((28, 28), dtype=np.float32))

                boxes_list.append(np.array(boxes, dtype=np.float32))
                labels_list.append(np.array(labels, dtype=np.int32))
                masks_list.append(np.array(masks, dtype=np.float32))

        if len(images_list) == 0:
            continue

        # Pad to batch size
        while len(images_list) < batch_size:
            images_list.append(images_list[0])
            boxes_list.append(boxes_list[0])
            labels_list.append(labels_list[0])
            masks_list.append(masks_list[0])

        # Convert to JAX arrays
        images = jnp.stack([jnp.array(img) for img in images_list])

        # Pad boxes/labels/masks to same length
        max_instances = max(len(b) for b in boxes_list)

        boxes_padded = []
        labels_padded = []
        masks_padded = []

        for boxes, labels, masks in zip(boxes_list, labels_list, masks_list):
            n_inst = len(boxes)
            if n_inst < max_instances:
                # Pad with zeros
                boxes = np.vstack([boxes, np.zeros((max_instances - n_inst, 4))])
                labels = np.concatenate([labels, np.zeros(max_instances - n_inst, dtype=np.int32)])
                masks = np.vstack([masks, np.zeros((max_instances - n_inst, 28, 28))])
            boxes_padded.append(boxes)
            labels_padded.append(labels)
            masks_padded.append(masks)

        boxes = jnp.array(np.stack(boxes_padded), dtype=jnp.float32)
        labels = jnp.array(np.stack(labels_padded), dtype=jnp.int32)
        masks = jnp.array(np.stack(masks_padded), dtype=jnp.float32)

        targets = MaskRCNNTargets(boxes=boxes, labels=labels, masks=masks)
        batches.append({"images": images, "targets": targets})

    return batches


def create_data_iterator(batches: list[dict], infinite: bool = True) -> Iterator[dict]:
    """Create an iterator over batches, optionally cycling infinitely."""
    import itertools

    if infinite:
        return itertools.cycle(iter(batches))
    return iter(batches)


def create_warmup_cosine_schedule(base_lr: float, warmup_steps: int, total_steps: int):
    """Create a learning rate schedule with warmup and cosine decay."""
    import optax

    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_steps)

    cosine_fn = optax.cosine_decay_schedule(init_value=base_lr, decay_steps=total_steps - warmup_steps, alpha=0.0)

    return optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])


def main():
    """Run Flax-based training."""
    args = parse_args()

    print("=" * 70)
    print("Mask R-CNN Training (Flax Infrastructure)")
    print("=" * 70)

    # Configuration
    print(f"\nDataset: {args.annotations}")
    print(f"Images: {args.images}")

    # Get number of classes
    print("\nDetecting number of classes from annotations...")
    num_classes = get_num_classes_from_coco(args.annotations)
    print(f"Number of classes: {num_classes}")

    # Create model config
    print("\nCreating model configuration...")
    model_config = MaskRCNNConfig(
        num_classes=num_classes,
        num_proposals=args.num_proposals,
        score_threshold=args.score_threshold,
        class_agnostic_bbox=args.class_agnostic_bbox,
        roi_pool_size=7,
        mask_pool_size=14,
        backbone={},
        fpn={},
        rpn={},
        detection_head={},
        mask_head={},
        anchor_generator={},
    )

    # Initialize model
    print("Initializing model...")
    model = MaskRCNN(model_config)

    # Load data
    print("\nLoading training data...")
    batches = load_coco_batch(args.annotations, args.images, batch_size=args.batch_size, max_images=args.max_images)
    print(f"Loaded {len(batches)} batches")

    if len(batches) == 0:
        print("❌ No data loaded! Check image paths.")
        return False

    # Create training config
    print("\nCreating training configuration...")

    # Learning rate schedule
    lr_schedule = create_warmup_cosine_schedule(base_lr=args.lr, warmup_steps=args.warmup_steps, total_steps=args.num_steps)

    config = ConfigDict(
        {
            "seed": args.seed,
            "model": {
                "module": model,
                "init_args": [batches[0]["images"]],
                "init_kwargs": {"training": True, "targets": batches[0]["targets"]},
                "train_kwargs": {"training": True},
            },
            "optimizer": {
                "learning_rate": lr_schedule,
                "weight_decay": args.weight_decay,
                "clip_norm": args.grad_clip,
                "beta1": 0.9,
                "beta2": 0.999,
            },
            "data": {
                "train_iter_fn": lambda: create_data_iterator(batches, infinite=True),
            },
            "training": {
                "num_steps": args.num_steps,
                "log_every_steps": args.log_every,
                "checkpoint_every_steps": args.checkpoint_every,
                "use_tqdm": True,
            },
            "checkpoint": {
                "dir": args.output_dir,
                "max_to_keep": args.max_checkpoints,
                "restore": True,
            },
        }
    )

    # Custom loss function that handles MaskRCNN's dict-based batch format
    def mask_rcnn_loss_fn(params, batch, model_state, rng, apply_fn):
        """Loss function for Mask R-CNN that unpacks batch dict."""
        images = batch["images"]
        targets = batch["targets"]

        variables = {"params": params}
        if model_state:
            variables.update(model_state)

        # Call model with training=True
        output = apply_fn(variables, images, training=True, targets=targets)

        # Extract loss and metrics
        loss = output["loss"]
        metrics = {k: v for k, v in output.items() if k != "loss"}

        return loss, metrics, model_state

    config.loss_fn = mask_rcnn_loss_fn

    print("\nTraining configuration:")
    print(f"  Steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr} (with warmup and cosine decay)")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Gradient clipping: {args.grad_clip}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Checkpoint dir: {args.output_dir}")

    # Run training
    print("\nStarting training loop...")
    print("-" * 70)

    result = train(config)

    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)

    # Print final metrics
    if result.history:
        final_metrics = result.history[-1]
        print("\nFinal metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")

    # Test inference
    print("\nTesting inference mode...")
    test_batch = batches[0]
    test_images = test_batch["images"]

    predictions = model.apply(
        {"params": result.state.params},
        test_images,
        training=False,
    )

    print("✅ Inference successful!")
    print(f"  Number of images: {len(predictions)}")
    print(f"  Detections per image: {predictions[0]['boxes'].shape[0]}")
    print(f"  Prediction keys: {list(predictions[0].keys())}")

    return True


if __name__ == "__main__":
    import traceback

    try:
        success = main()
        if success:
            print("\n" + "=" * 70)
            print("🎉 TRAINING SUCCESSFUL!")
            print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TRAINING FAILED: {e}")
        print("=" * 70)
        traceback.print_exc()
