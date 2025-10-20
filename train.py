#!/usr/bin/env python3
"""Convenient training script for Mask R-CNN.

Usage:
    python train.py --annotations /path/to/train.json --images /path/to/images

    # With custom config
    python train.py --annotations /path/to/train.json --images /path/to/images \
        --num-proposals 1000 --batch-size 4 --epochs 10 --lr 1e-4

    # CMR dataset (default paths)
    python train.py --cmr
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from detectrax.data.coco_utils import get_num_classes_from_coco
from detectrax.models.detectors.mask_rcnn import MaskRCNN, MaskRCNNConfig, MaskRCNNTargets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on COCO dataset")

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
    parser.set_defaults(class_agnostic_bbox=True)  # Default: True (class-agnostic)

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to load (default: all)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
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

    import numpy as np
    from PIL import Image

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
        batches.append((images, targets))

    return batches


def create_train_state(model, rng, dummy_input, dummy_targets, learning_rate=1e-5, grad_clip=1.0):
    """Initialize training state."""
    # Initialize model
    variables = model.init(rng, dummy_input, training=True, targets=dummy_targets)
    params = variables["params"]

    # Create optimizer with gradient clipping to prevent NaN
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),  # Clip gradients to prevent explosions
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params)

    return params, opt_state, optimizer


def train_step(model, params, opt_state, optimizer, images, targets):
    """Single training step."""

    def loss_fn(params):
        losses = model.apply(
            {"params": params},
            images,
            training=True,
            targets=targets,
        )
        return losses["loss"], losses

    (loss, losses_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Check for NaN gradients
    grad_norm = optax.global_norm(grads)
    if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
        print(f"  WARNING: Gradient norm is {grad_norm}, skipping update")
        return params, opt_state, loss, losses_dict

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, losses_dict


def main():
    """Run training."""
    args = parse_args()

    print("=" * 70)
    print("Mask R-CNN Training")
    print("=" * 70)

    # Configuration
    print(f"\nDataset: {args.annotations}")
    print(f"Images: {args.images}")

    # Get number of classes
    print("\nDetecting number of classes from annotations...")
    num_classes = get_num_classes_from_coco(args.annotations)
    print(f"Number of classes: {num_classes}")

    # Create config
    print("\nCreating model configuration...")
    config = MaskRCNNConfig(
        num_classes=num_classes,
        num_proposals=args.num_proposals,
        score_threshold=args.score_threshold,
        class_agnostic_bbox=args.class_agnostic_bbox,  # Default: True (class-agnostic bbox)
        roi_pool_size=7,
        mask_pool_size=14,
        backbone={},
        fpn={},
        rpn={},
        detection_head={},
        mask_head={},
        anchor_generator={},
    )

    print(f"Config: num_proposals={args.num_proposals}, lr={args.lr}, batch_size={args.batch_size}")

    # Initialize model
    print("Initializing model...")
    model = MaskRCNN(config)

    # Load data
    print("\nLoading training data...")
    batches = load_coco_batch(args.annotations, args.images, batch_size=args.batch_size, max_images=args.max_images)
    print(f"Loaded {len(batches)} batches")

    if len(batches) == 0:
        print("❌ No data loaded! Check image paths.")
        return False

    # Initialize training state
    print("\nInitializing training state...")
    rng = jax.random.PRNGKey(args.seed)
    images_init, targets_init = batches[0]
    params, opt_state, optimizer = create_train_state(model, rng, images_init, targets_init, learning_rate=args.lr, grad_clip=args.grad_clip)

    # Training loop
    print("\nStarting training loop...")
    print("-" * 70)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_losses = []

        for batch_idx, (images, targets) in enumerate(tqdm(batches, desc=f"Epoch {epoch + 1}")):
            params, opt_state, loss, losses_dict = train_step(model, params, opt_state, optimizer, images, targets)

            epoch_losses.append(float(loss))

            if batch_idx == 0 or (batch_idx + 1) % max(1, len(batches) // 2) == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(batches)}: "
                    f"Loss={loss:.4f} | "
                    f"RPN_cls={losses_dict['rpn_cls']:.4f} | "
                    f"Det_cls={losses_dict['det_cls']:.4f} | "
                    f"Mask={losses_dict['mask']:.4f}"
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)

    # Test inference
    print("\nTesting inference mode...")
    test_images, _ = batches[0]
    predictions = model.apply(
        {"params": params},
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
