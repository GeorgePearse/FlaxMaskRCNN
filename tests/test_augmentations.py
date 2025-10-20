import numpy as np
import tensorflow as tf

from detectax.data.augmentations import AugmentationConfig, augment_example


def test_horizontal_flip_updates_boxes_and_masks() -> None:
    config = AugmentationConfig(
        flip_prob=1.0,
        scale_min=1.0,
        scale_max=1.0,
        color_jitter_prob=0.0,
    )

    image = tf.cast(tf.reshape(tf.range(4 * 8 * 3), (4, 8, 3)), tf.uint8)
    boxes = tf.constant([[1.0, 0.0, 6.0, 3.0]], dtype=tf.float32)
    mask = np.zeros((1, 4, 8), dtype=np.uint8)
    mask[0, :, :3] = 1
    masks = tf.constant(mask)

    augmented = augment_example(
        {"image": image, "boxes": boxes, "masks": masks},
        config=config,
        seed=tf.constant([0, 42], dtype=tf.int32),
    )

    expected_boxes = np.array([[2.0, 0.0, 7.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(
        augmented["boxes"].numpy(),
        expected_boxes,
        rtol=1e-5,
        atol=1e-5,
    )

    flipped_mask = augmented["masks"].numpy()
    assert flipped_mask.shape == (1, 4, 8)
    np.testing.assert_array_equal(flipped_mask[0, :, -3:], 1)
    np.testing.assert_array_equal(flipped_mask[0, :, :5], 0)


def test_scale_jitter_resizes_image_and_boxes() -> None:
    config = AugmentationConfig(
        flip_prob=0.0,
        scale_min=1.5,
        scale_max=1.5,
        color_jitter_prob=0.0,
    )

    image = tf.ones((4, 4, 3), dtype=tf.uint8) * 100
    boxes = tf.constant([[1.0, 1.0, 3.0, 3.0]], dtype=tf.float32)
    masks = tf.ones((1, 4, 4), dtype=tf.uint8)

    augmented = augment_example(
        {"image": image, "boxes": boxes, "masks": masks},
        config=config,
        seed=tf.constant([1, 7], dtype=tf.int32),
    )

    aug_image = augmented["image"]
    assert aug_image.shape == (6, 6, 3)

    expected_boxes = np.array([[1.5, 1.5, 4.5, 4.5]], dtype=np.float32)
    np.testing.assert_allclose(
        augmented["boxes"].numpy(),
        expected_boxes,
        rtol=1e-5,
        atol=1e-5,
    )

    aug_masks = augmented["masks"].numpy()
    assert aug_masks.shape == (1, 6, 6)
    assert np.all(aug_masks == 1)


def test_normalization_matches_imagenet_stats() -> None:
    config = AugmentationConfig(
        flip_prob=0.0,
        scale_min=1.0,
        scale_max=1.0,
        color_jitter_prob=0.0,
        assume_float_range01=True,
    )

    image = tf.ones((2, 2, 3), dtype=tf.float32) * 0.5
    augmented = augment_example({"image": image}, config=config)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    per_channel = (0.5 - mean) / std
    expected = np.broadcast_to(per_channel, (2, 2, 3))
    np.testing.assert_allclose(
        augmented["image"].numpy(),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_deterministic_with_seed() -> None:
    config = AugmentationConfig(
        flip_prob=1.0,
        scale_min=0.8,
        scale_max=1.2,
        color_jitter_prob=1.0,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    )

    image = tf.cast(tf.reshape(tf.range(32 * 32 * 3), (32, 32, 3)), tf.uint8)

    seed_a = tf.constant([3, 5], dtype=tf.int32)
    seed_b = tf.constant([11, 17], dtype=tf.int32)

    augmented_a1 = augment_example({"image": image}, config=config, seed=seed_a)
    augmented_a2 = augment_example({"image": image}, config=config, seed=seed_a)
    augmented_b = augment_example({"image": image}, config=config, seed=seed_b)

    image_a1 = augmented_a1["image"].numpy()
    image_a2 = augmented_a2["image"].numpy()
    image_b = augmented_b["image"].numpy()

    np.testing.assert_allclose(
        image_a1,
        image_a2,
        rtol=1e-6,
        atol=1e-6,
    )

    if image_a1.shape == image_b.shape:
        assert not np.allclose(image_a1, image_b, atol=1e-6)
    else:
        assert image_a1.shape != image_b.shape
