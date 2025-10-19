# Scenic Architecture Notes

Research notes from studying Google Scenic for Mask R-CNN implementation.

## Overview

Scenic provides a framework for vision models in JAX/Flax with:
- Base model classes for different tasks (classification, segmentation, detection)
- Training infrastructure (distributed training, metrics, checkpointing)
- Model library (layers, matchers, backbones)
- Example projects (DETR, Deformable DETR, OWL-ViT, Segment Anything)

## Key Components

### Base Models (`scenic/model_lib/base_models/`)

- `base_model.py` - Abstract base class for all models
- `classification_model.py` - Classification task base
- `segmentation_model.py` - Segmentation task base
- `encoder_decoder_model.py` - Encoder-decoder architectures
- `model_utils.py` - Utilities (metrics, weights, distributed training)

### DETR Detection Baseline (`scenic/projects/baselines/detr/`)

Key files for understanding detection in Scenic:
- `model.py` - DETR model implementation using Flax nn.Module
- `detr_base_model.py` - Base detection model class
- `input_pipeline_detection.py` - COCO data loading
- `trainer.py` - Training loop with distributed training
- `train_utils.py` - Training utilities
- `transforms.py` - Data augmentation

### Model Library (`scenic/model_lib/`)

- `layers/` - Common layers (attention, masking, etc.)
- `matchers/` - Hungarian matching for detection
- `base_models/` - Task-specific base classes

## Scenic Patterns for Mask R-CNN

### 1. Model Definition

```python
class MaskRCNN(nn.Module):
    config: ml_collections.ConfigDict

    @nn.compact
    def __call__(self, x, train=False):
        # Functional forward pass
        # Use self.param() for learnable parameters
        # Return dict with predictions and losses
```

### 2. Base Model Integration

Extend Scenic's base model (likely classification or encoder-decoder):
- Implement `build_flax_model()` - returns the Flax nn.Module
- Implement `loss_function()` - computes loss from model output
- Implement `get_metrics_fn()` - returns metric computation functions
- Implement `default_flax_model_config()` - default configuration

### 3. Training Infrastructure

Scenic provides:
- Distributed training with `jax.pmap`
- Metrics aggregation with `psum_metric_normalizer`
- Checkpointing with orbax-checkpoint
- Learning rate schedules
- Mixed precision training

### 4. Data Pipeline

Pattern from DETR:
- Use `tf.data` API for efficient data loading
- Implement COCO parsing
- Apply data augmentation transforms
- Batch and prefetch data

## Implementation Strategy for FlaxMaskRCNN

### Phase 1: Use Scenic Components

1. **Backbone**: Use Scenic's ResNet from `scenic/projects/baselines/resnet.py`
2. **FPN**: Implement FPN as Flax nn.Module following Scenic patterns
3. **Base Model**: Extend `classification_model.py` or create custom base
4. **Training**: Use DETR's trainer as template

### Phase 2: Port from visdet Reference

Reference PyTorch visdet code while implementing:
1. RPN head architecture
2. RoI operations (align/pool)
3. Detection head (bbox + classification)
4. Mask head (FCN)
5. Loss functions

### Phase 3: Adapt DETR Data Pipeline

Modify `input_pipeline_detection.py`:
- Keep COCO loading logic
- Adjust for two-stage detection (RPN proposals)
- Add mask ground truth handling
- Adapt augmentations for Mask R-CNN

## Key Differences: Scenic vs PyTorch

### Scenic/JAX Patterns

- **Functional**: No mutable state, explicit PRNG keys
- **Pytrees**: Parameters are nested dicts/tuples
- **Transforms**: Use jax.jit, jax.vmap, jax.pmap for performance
- **Training State**: Use Flax TrainState or Scenic's train_state
- **Distributed**: Built-in pmap support for multi-device training

### visdet/PyTorch Patterns

- **Imperative**: Mutable module state
- **nn.Module**: Stateful modules with forward()
- **Autograd**: Automatic differentiation
- **DDP/FSDP**: External distributed training wrappers

## Next Steps

1. ✅ Research Scenic architecture - COMPLETE
2. ⏳ Implement FPN neck using Scenic patterns
3. ⏳ Integrate Scenic ResNet backbone
4. ⏳ Implement RPN following visdet architecture
5. ⏳ Port data pipeline from DETR baseline
6. ⏳ Implement Mask R-CNN detector

## Resources

- Scenic GitHub: https://github.com/google-research/scenic
- DETR paper: https://arxiv.org/abs/2005.12872
- Mask R-CNN paper: https://arxiv.org/abs/1703.06870
- JAX documentation: https://jax.readthedocs.io/
- Flax documentation: https://flax.readthedocs.io/
