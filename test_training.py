"""Minimal training test script for Mask R-CNN."""

import jax
import jax.numpy as jnp

from detectax.models.detectors.mask_rcnn import MaskRCNN, MaskRCNNConfig, MaskRCNNTargets


def create_minimal_config() -> MaskRCNNConfig:
    """Create minimal config for testing."""
    return MaskRCNNConfig(
        num_classes=3,  # CMR dataset has 3 classes
        num_proposals=100,
        score_threshold=0.05,
        class_agnostic_bbox=True,
        roi_pool_size=7,
        mask_pool_size=14,
        backbone={},
        fpn={},
        rpn={},
        detection_head={},
        mask_head={},
        anchor_generator={},
    )


def create_dummy_batch(batch_size: int = 2, num_classes: int = 3):
    """Create dummy batch for smoke testing."""
    # Dummy images
    images = jnp.ones((batch_size, 512, 512, 3), dtype=jnp.float32)

    # Dummy GT boxes (2 instances per image)
    boxes = jnp.array(
        [
            [[100, 100, 200, 200], [250, 250, 350, 350]],
            [[50, 50, 150, 150], [300, 300, 400, 400]],
        ],
        dtype=jnp.float32,
    )

    # Dummy GT labels (class 1 and 2)
    labels = jnp.array([[1, 2], [2, 1]], dtype=jnp.int32)

    # Dummy GT masks (28x28 binary masks)
    masks = jnp.ones((batch_size, 2, 28, 28), dtype=jnp.float32)

    targets = MaskRCNNTargets(
        boxes=boxes,
        labels=labels,
        masks=masks,
    )

    return images, targets


def test_training_step():
    """Test a single training step."""
    print("Initializing model...")
    config = create_minimal_config()
    model = MaskRCNN(config)

    print("Creating dummy batch...")
    images, targets = create_dummy_batch(batch_size=2, num_classes=3)

    print("Initializing parameters...")
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, images, training=True, targets=targets)
    params = variables["params"]

    print("Testing forward pass (training mode)...")

    def loss_fn(params):
        losses = model.apply(
            {"params": params},
            images,
            training=True,
            targets=targets,
        )
        return losses["loss"], losses

    # Compute loss and gradients
    (loss, losses_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    print("\n✅ Training step successful!")
    print(f"Loss: {loss:.4f}")
    print(f"Loss components: {list(losses_dict.keys())}")
    print(f"  RPN cls: {losses_dict['rpn_cls']:.4f}")
    print(f"  RPN reg: {losses_dict['rpn_reg']:.4f}")
    print(f"  Det cls: {losses_dict['det_cls']:.4f}")
    print(f"  Det reg: {losses_dict['det_reg']:.4f}")
    print(f"  Mask: {losses_dict['mask']:.4f}")
    print("\nGradient norms computed successfully")

    # Test inference mode
    print("\nTesting forward pass (inference mode)...")
    predictions = model.apply(
        {"params": params},
        images,
        training=False,
    )

    print("✅ Inference successful!")
    print(f"Predictions type: {type(predictions)}")
    print(f"Number of images: {len(predictions)}")
    print(f"First prediction keys: {list(predictions[0].keys())}")
    print(f"Number of detections (image 0): {predictions[0]['boxes'].shape[0]}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Mask R-CNN Training Smoke Test")
    print("=" * 60)

    try:
        success = test_training_step()
        if success:
            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED - Training pipeline works!")
            print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
