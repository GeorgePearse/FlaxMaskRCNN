# Detectax: Detection with JAX

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35-orange.svg)](https://github.com/google/jax)

⚠️ **Experimental hobby project** - This is very much a work-in-progress mess-around repo, largely AI-generated code. Not production-ready by any means!

```python
import detectrax as dax

# Load model and run inference
model = dax.MaskRCNN(num_classes=80)
predictions = model(images)
```

## Overview

Detectax is an attempt to implement Mask R-CNN for object detection and instance segmentation using JAX/Flax. It's a learning/exploration project that aims to:

- **Build Mask R-CNN components**: Backbone, FPN, RPN, detection & mask heads
- **Learn JAX patterns**: Functional programming, jit compilation, pmap/pjit
- **Type-safe code**: Full type annotations with jaxtyping
- **Test as we go**: Unit tests for components we build
- **Document the journey**: Notes on architecture and implementation decisions

## Quick Start

Get started with Detectax in 3 commands:

```bash
# Install dependencies
uv sync

# Run tests to verify installation
uv run pytest -v

# Train on COCO dataset (coming soon)
uv run python -m detectrax.training.train --config configs/mask_rcnn_r50_fpn.yaml
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FlaxMaskRCNN.git
cd FlaxMaskRCNN

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Development Installation

For development with type checking and testing:

```bash
# Install with all dev dependencies
uv sync --all-extras

# Verify installation
uv run pytest -v
uv run pyright
```

### GPU Support

JAX with CUDA 12 support for GPU training:

```bash
# Install JAX with CUDA support
pip install --upgrade "jax[cuda12]"
```

## Usage

### Training

```bash
# Train Mask R-CNN on COCO dataset
uv run python -m detectrax.training.train \\
  --config configs/mask_rcnn_r50_fpn.yaml \\
  --data_dir <path-to-coco> \\
  --output_dir ./runs/experiment1
```

### Evaluation

```bash
# Evaluate model on COCO val set
uv run python -m detectrax.evaluation.evaluate \\
  --config configs/mask_rcnn_r50_fpn.yaml \\
  --checkpoint <path-to-checkpoint> \\
  --data_dir <path-to-coco>
```

### Configuration

Configuration via YAML files using ml_collections:

```yaml
# configs/mask_rcnn_r50_fpn.yaml
model:
  backbone: resnet50
  num_classes: 80
  fpn_channels: 256

training:
  batch_size: 16
  learning_rate: 0.02
  num_epochs: 12
  optimizer: sgd

data:
  dataset: coco
  train_annotation: <path-to-train-annotations>
  val_annotation: <path-to-val-annotations>
  image_dir: <path-to-images>
```

See [docs/datasets.md](docs/datasets.md) for detailed dataset configuration.

## Project Structure

```
detectrax/
├── detectrax/
│   ├── models/
│   │   ├── backbones/      # Feature extractors (ResNet, ViT)
│   │   ├── necks/          # FPN implementation
│   │   ├── heads/          # RPN, detection, mask heads
│   │   ├── detectors/      # Complete Mask R-CNN model
│   │   └── layers/         # RoI Align, custom layers
│   ├── data/               # Data loading and augmentation
│   ├── configs/            # YAML configuration files
│   ├── evaluation/         # COCO evaluation metrics
│   ├── training/           # Training loops, optimizers
│   └── utils/              # Utilities (checkpointing, logging)
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
├── reference/              # PyTorch reference code (visdet)
├── scenic_repo/            # Google Scenic submodule
├── pyproject.toml          # Project dependencies
└── README.md
```

## Architecture

Mask R-CNN pipeline for object detection and instance segmentation:

1. **Backbone**: ResNet or Vision Transformer for feature extraction
2. **FPN**: Feature Pyramid Network for multi-scale features
3. **RPN**: Region Proposal Network generates object proposals
4. **RoI Align**: Extract features for each proposal
5. **Detection Head**: Classification + bounding box regression
6. **Mask Head**: Instance segmentation masks

For detailed architecture documentation, see [docs/architecture.md](docs/architecture.md).

## Performance Benchmarks

Target performance on COCO 2017 validation set:

| Backbone | mAP | AP50 | AP75 | FPS (V100) | Status |
|----------|-----|------|------|------------|--------|
| ResNet-50 | ~38 | ~58 | ~41 | TBD | In development |
| ResNet-101 | ~40 | ~60 | ~43 | TBD | Planned |

See [docs/benchmarks.md](docs/benchmarks.md) for detailed benchmark results and reproducibility information.

## Development

### Code Quality

```bash
# Run linting
uv run ruff check .

# Format code
uv run black .

# Type checking
uv run pyright

# Run all checks
uv run pytest && uv run pyright && uv run ruff check .
```

### Testing

```bash
# Run all tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=detectrax --cov-report=html

# Run specific test file
uv run pytest tests/test_fpn.py -v
```

### Spec-Driven Development

Detectax uses [GitHub Spec Kit](https://github.com/github/spec-kit) for spec-driven development with AI assistants. This enables a structured workflow:

1. **Define**: Create specifications that focus on what the feature does (not how)
2. **Plan**: Break down specs into implementation plans
3. **Build**: Generate tasks and implement with AI assistance
4. **Validate**: Ensure implementations match specifications

**Available slash commands:**

```bash
/speckit.constitution   # Establish project principles
/speckit.specify        # Create feature specification
/speckit.plan          # Create implementation plan
/speckit.tasks         # Generate actionable tasks
/speckit.implement     # Execute implementation

# Optional enhancement commands
/speckit.clarify       # Ask structured questions to de-risk ambiguous areas
/speckit.analyze       # Cross-artifact consistency & alignment report
/speckit.checklist     # Generate quality checklists
```

**Example workflow:**

```bash
# Start with a natural language description
/speckit.specify Add support for training on custom COCO-format datasets

# Review and refine the generated specification
# Then create an implementation plan
/speckit.plan

# Generate actionable tasks
/speckit.tasks

# Execute implementation
/speckit.implement
```

All specifications and plans are stored in `.specify/` and tracked in git. The `.claude/` directory contains agent-specific configurations and is excluded from version control.

## Roadmap

**Current Status**: Phase 2 - Core Model Implementation (40% complete)

### Near Term (v0.1.0)
- ✅ FPN and backbone implementation
- ✅ RPN and RoI Align
- 🚧 Detection and mask heads
- 🚧 Complete Mask R-CNN detector
- 📅 COCO data pipeline
- 📅 Training infrastructure with Scenic

### Medium Term (v0.2.0)
- Multi-GPU distributed training
- COCO benchmark results
- Additional backbones (ViT, Swin)
- TensorBoard/W&B integration

### Long Term (v1.0.0)
- Stable API
- Deployment examples (ONNX, TensorRT)
- Advanced augmentations
- Model zoo with pretrained weights

See [TODO.md](TODO.md) for detailed roadmap and task breakdown.

## Key Features

### JAX/Flax Benefits
- **Functional Programming**: Pure functions, no hidden state
- **High Performance**: JIT compilation with XLA
- **Easy Parallelization**: pmap/pjit for multi-GPU/TPU
- **Type Safety**: jaxtyping for array shape annotations
- **Reproducibility**: Explicit PRNGKey management

Want to learn more about JAX? See [docs/why_jax.md](docs/why_jax.md).

### Development Goals
- Distributed training with JAX's pmap/pjit (planned)
- Checkpoint saving with orbax-checkpoint (planned)
- COCO evaluation metrics (AP, AP50, AP75, AR) (planned)
- Progress tracking with tqdm
- Configuration management with ml_collections (planned)

## Documentation

- **[Architecture Guide](docs/architecture.md)**: Detailed model architecture
- **[Dataset Configuration](docs/datasets.md)**: COCO format and custom datasets
- **[Performance Benchmarks](docs/benchmarks.md)**: Speed and accuracy metrics
- **[Why JAX?](docs/why_jax.md)**: JAX ecosystem and benefits
- **[Project Roadmap](TODO.md)**: Development status and tasks

## Dependencies

Core dependencies:
- **JAX** 0.4.35 - High-performance numerical computing
- **Flax** 0.10.2 - Neural network library for JAX
- **Scenic** - Training infrastructure from [google-research/scenic](https://github.com/google-research/scenic)
- **Optax** 0.2.4 - Gradient processing and optimization
- **jaxtyping** 0.2.34 - Type annotations for arrays
- **TensorFlow** 2.18.0 - Data loading pipeline
- **pycocotools** 2.0.8 - COCO evaluation metrics

See [pyproject.toml](pyproject.toml) for complete dependency list.

## Contributing

Contributions are welcome! Priority areas:

1. **Testing**: Write unit tests for existing components
2. **Documentation**: Add tutorials and examples
3. **Benchmarking**: Run and report performance metrics
4. **Features**: Implement items from [TODO.md](TODO.md)

Please see [CLAUDE.md](CLAUDE.md) for development guidelines and best practices.

## Reference Code

The `reference/` directory contains PyTorch code from visdet for architecture reference. This helps ensure mathematical correctness when porting to JAX/Flax.

## Resources

### JAX Ecosystem
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Scenic Repository](https://github.com/google-research/scenic) - Training infrastructure
- [Haliax](https://github.com/marin-community/haliax) - Named tensor library for JAX
- [Mixed Precision for JAX](https://github.com/Data-Science-in-Mechanical-Engineering/mixed_precision_for_JAX) - Mixed precision training utilities

### Papers
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Feature Pyramid Networks Paper](https://arxiv.org/abs/1612.03144)

## License

See [LICENSE](LICENSE) file for details.

## Citation

If you use Detectax in your research, please cite:

```bibtex
@software{detectrax2025,
  title={Detectax: Experimental Mask R-CNN in JAX/Flax},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/FlaxMaskRCNN},
  note={Experimental hobby project - not production-ready}
}
```

---

**Status**: Experimental - Very early stage, mostly AI-generated code, expect bugs and incomplete features
**Python**: 3.12+ | **JAX**: 0.4.35+ | **License**: Apache 2.0
