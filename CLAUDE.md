# FlaxMaskRCNN Project Guidelines

## Purpose

Production-ready Mask R-CNN implementation in JAX/Flax using Google Scenic's training infrastructure.

## Technology Stack

- **Language**: Python 3.12
- **ML Framework**: JAX 0.4.35, Flax 0.10.2
- **Training Infrastructure**: Google Scenic (to be integrated from https://github.com/google-research/scenic)
- **Optimization**: Optax 0.2.4
- **Data Loading**: TensorFlow 2.18.0, tensorflow-datasets 4.9.7
- **Evaluation**: pycocotools 2.0.8
- **Type Checking**: Pyright, mypy, jaxtyping
- **Testing**: Pytest
- **Dependency Management**: uv

## Project Structure

```
flax_mask_rcnn/
├── models/
│   ├── backbones/       # Feature extractors (ResNet, ViT from Scenic)
│   ├── necks/           # FPN implementation
│   ├── heads/           # RPN, detection, mask heads
│   └── detectors/       # Complete Mask R-CNN model
├── data/                # Data loading and augmentation
├── configs/             # YAML configuration files
├── evaluation/          # COCO evaluation metrics
├── training/            # Training loops, optimizers, schedules
└── utils/               # Utilities (checkpointing, logging, etc.)
tests/                   # Unit and integration tests
reference/               # PyTorch visdet reference code (for architecture understanding)
```

## Key Principles

### 1. JAX/Flax Best Practices

- **Functional Programming**: All model code must be functional and pure
- **Explicit State**: Use Flax's Module system, no hidden mutable state
- **Pytrees**: Understand and use JAX pytrees for parameters and state
- **Transformations**: Use jax.jit, jax.vmap, jax.pmap appropriately
- **Random Keys**: Always pass and split PRNGKeys explicitly
- **No Side Effects**: Avoid print statements inside jitted functions

### 2. Scenic Integration

- **Follow Scenic Patterns**: Study and follow Scenic's architectural patterns
- **Use Scenic Base Classes**: Extend Scenic's base model and trainer classes
- **Reuse Components**: Leverage Scenic's existing backbones, FPN, and utilities
- **Training Infrastructure**: Use Scenic's distributed training, checkpointing, and metrics
- **Configuration**: Use ml_collections.ConfigDict for all configs

### 3. Type Safety

- **Complete Type Annotations**: All functions must have full type hints
- **jaxtyping**: Use jaxtyping for array shape annotations (Float[Array, "batch height width channels"])
- **No `Any` Types**: Avoid Any unless absolutely necessary and documented
- **Strict Mode**: Code must pass pyright in strict mode
- **Runtime Checks**: Use chex for runtime shape/type assertions during development

### 4. Code Quality

- **Never Simulate**: Never create fake losses or simulated training metrics
- **Full Implementation**: Always implement complete functionality, no simplified versions
- **Delete and Restart**: If code doesn't work, delete it and start fresh instead of patching
- **No System Path Hacks**: Fix underlying import issues instead of appending to sys.path
- **Line Length**: 150 characters maximum
- **Imports**: Use absolute imports within package

### 5. Testing

- **Test Data**: Use CMR COCO dataset for testing
  - Train: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
  - Val: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`
  - Images: `/home/georgepearse/data/images`
- **Unit Tests**: Test individual components (FPN, RPN, heads) in isolation
- **Integration Tests**: Test full model forward pass on small batches
- **Shape Tests**: Verify all tensor shapes throughout the model
- **Class Count**: Always verify annotation class counts match model architecture

### 6. Training

- **Progress Bars**: Always use tqdm for long-running operations
- **Structured Logging**: Use Python's logging module with clear format
- **Checkpointing**: Save checkpoints regularly using orbax-checkpoint
- **Reproducibility**: Set random seeds for reproducibility
- **Configuration**: All hyperparameters must be in YAML configs validated with ml_collections

### 7. Data Pipeline

- **COCO Format**: Follow COCO JSON annotation format
- **Augmentation**: Implement augmentations compatible with JAX (or use TF ops)
- **Validation**: Validate data formats and annotations early in the pipeline
- **Efficiency**: Use tf.data or grain for efficient data loading with prefetching
- **Mask Formats**: Ensure mask formats are consistent throughout (binary masks as arrays)

### 8. Reference Code Usage

- **Keep Reference Visible**: PyTorch visdet code in `reference/` directory provides architecture details
- **Port, Don't Copy**: Understand PyTorch code, then implement in JAX/Flax idiomatically
- **Document Differences**: Note where JAX implementation differs from PyTorch reference
- **Architecture Fidelity**: Match the mathematical operations, not the imperative style

## Common Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest -v

# Run type checking
uv run pyright

# Run linting
uv run ruff check .

# Format code
uv run black .

# Run training (to be implemented)
uv run python -m flax_mask_rcnn.training.train --config configs/mask_rcnn_r50_fpn.yaml
```

## Implementation Phases

1. **Phase 1**: Research Scenic, copy reference code from visdet
2. **Phase 2**: Implement core model (backbone, FPN, RPN, heads)
3. **Phase 3**: Port data pipeline from visdet
4. **Phase 4**: Implement training infrastructure with Scenic
5. **Phase 5**: Add evaluation and testing
6. **Phase 6**: Documentation and finalization

## Important Notes

- **Scenic Repository**: Clone https://github.com/google-research/scenic into the project or reference externally
- **GPU Support**: JAX with CUDA 12 support for GPU training
- **Distributed Training**: Use JAX's pmap/pjit for multi-GPU training
- **No PyTorch**: This is a pure JAX/Flax implementation, PyTorch visdet code is reference only

## Resources

- JAX Documentation: https://jax.readthedocs.io/
- Flax Documentation: https://flax.readthedocs.io/
- Scenic Repository: https://github.com/google-research/scenic
- visdet Reference: `/home/georgepearse/core/machine_learning/packages/visdet`

---

*For general machine learning guidelines, see the parent machine_learning/CLAUDE.md*
*For general repository guidelines, see the root CLAUDE.md*
