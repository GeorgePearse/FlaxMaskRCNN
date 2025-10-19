# Detectax TODO & Roadmap

## Project Status

**Current Phase**: Phase 2 - Core Model Implementation (40% complete)
**Stability**: Alpha - API may change
**Last Updated**: 2025-10-19

## Implementation Phases

### ✅ Phase 1: Research & Setup (Complete)
- [x] Research Scenic architecture patterns
- [x] Copy PyTorch reference code from visdet
- [x] Set up project structure with uv
- [x] Configure type checking (pyright, mypy, jaxtyping)
- [x] Set up testing infrastructure (pytest)

### 🚧 Phase 2: Core Model Implementation (In Progress - 40%)

#### Completed Components
- [x] Feature Pyramid Network (FPN) - `detectax/models/necks/fpn.py`
- [x] ResNet backbone - `detectax/models/backbones/resnet.py`
- [x] Region Proposal Network (RPN) head - `detectax/models/heads/rpn.py`
- [x] RoI Align layer - `detectax/models/layers/roi_align.py`
- [x] FPN unit tests - `tests/test_fpn.py`

#### In Progress
- [ ] **Detection Head** (`detectax/models/heads/bbox_head.py`)
  - [ ] Box regression and classification
  - [ ] Cascade refinement (optional)
  - [ ] Unit tests

- [ ] **Mask Head** (`detectax/models/heads/mask_head.py`)
  - [ ] Convolutional mask predictor
  - [ ] Per-class mask outputs
  - [ ] Unit tests

- [ ] **Complete Mask R-CNN Detector** (`detectax/models/detectors/mask_rcnn.py`)
  - [ ] Integrate all components (backbone, FPN, RPN, heads)
  - [ ] Forward pass implementation
  - [ ] Loss computation
  - [ ] Inference pipeline with NMS
  - [ ] Integration tests

- [ ] **Additional Backbones**
  - [ ] Vision Transformer (ViT) integration from Scenic
  - [ ] Swin Transformer (future)

### 📦 Phase 3: Data Pipeline (Not Started)

- [ ] **COCO Dataset Loader** (`detectax/data/coco.py`)
  - [ ] Load annotations in COCO format
  - [ ] Image loading and preprocessing
  - [ ] Class count verification
  - [ ] Data validation utilities

- [ ] **Data Augmentation** (`detectax/data/augmentation.py`)
  - [ ] Random flip, crop, resize
  - [ ] Color jitter
  - [ ] Mosaic augmentation (optional)
  - [ ] Mixup (optional)

- [ ] **Data Pipeline Integration**
  - [ ] tf.data pipeline or grain integration
  - [ ] Efficient batching and prefetching
  - [ ] Multi-worker data loading

### 🎯 Phase 4: Training Infrastructure (Not Started)

- [ ] **Training Loop** (`detectax/training/train.py`)
  - [ ] Integrate Scenic trainer
  - [ ] Multi-GPU support with pmap/pjit
  - [ ] Gradient accumulation
  - [ ] Mixed precision training (FP16)

- [ ] **Optimization**
  - [ ] SGD with momentum
  - [ ] Adam optimizer
  - [ ] Learning rate schedules (step, cosine, warmup)
  - [ ] Gradient clipping

- [ ] **Checkpointing** (`detectax/training/checkpoint.py`)
  - [ ] Save/load with orbax-checkpoint
  - [ ] Best model tracking
  - [ ] Resume from checkpoint

- [ ] **Logging & Monitoring**
  - [ ] TensorBoard integration
  - [ ] Weights & Biases integration (optional)
  - [ ] Progress bars with tqdm

### 📊 Phase 5: Evaluation & Testing (Not Started)

- [ ] **COCO Evaluation** (`detectax/evaluation/coco_eval.py`)
  - [ ] Integrate pycocotools
  - [ ] Compute mAP, AP50, AP75, etc.
  - [ ] Per-category metrics
  - [ ] Visualization utilities

- [ ] **Testing**
  - [ ] Unit tests for all components (80%+ coverage)
  - [ ] Integration tests for full pipeline
  - [ ] Benchmarking scripts
  - [ ] Test on CMR dataset

### 📚 Phase 6: Documentation & Polish (Partially Complete)

- [x] Project README
- [x] Architecture documentation
- [x] Dataset configuration guide
- [x] Why JAX documentation
- [ ] API documentation (auto-generated)
- [ ] Training tutorials
- [ ] Inference examples
- [ ] Jupyter notebooks
- [ ] CONTRIBUTING.md guide

## High-Priority Tasks (Next Sprint)

### 1. Complete Detection Head (Highest Priority)
**Assignee**: TBD
**Effort**: 4-6 hours
**Dependencies**: None (RPN already implemented)

- Implement box regression and classification layers
- Add smooth L1 loss for bounding boxes
- Add cross-entropy loss for classification
- Write unit tests for forward pass and loss computation
- Verify output shapes match expected dimensions

### 2. Implement Mask Head
**Assignee**: TBD
**Effort**: 4-6 hours
**Dependencies**: Detection head

- Implement convolutional mask predictor (4 conv layers + upsampling)
- Add per-class mask outputs (28x28 per class)
- Implement binary cross-entropy loss
- Write unit tests
- Verify mask predictions align with RoI Align outputs

### 3. Build Complete Mask R-CNN Detector
**Assignee**: TBD
**Effort**: 6-8 hours
**Dependencies**: Detection head, Mask head

- Integrate backbone, FPN, RPN, detection head, mask head
- Implement full forward pass (training and inference modes)
- Implement total loss computation (RPN + detection + mask)
- Add NMS post-processing for inference
- Write integration tests with synthetic data
- Test on a small subset of CMR dataset

### 4. Implement COCO Data Loader
**Assignee**: TBD
**Effort**: 6-8 hours
**Dependencies**: None

- Parse COCO JSON annotations
- Load and preprocess images
- Create batching logic compatible with JAX
- Add basic augmentations (flip, resize)
- Verify annotation format and class counts
- Test with CMR dataset

### 5. Set Up Basic Training Loop
**Assignee**: TBD
**Effort**: 8-10 hours
**Dependencies**: Complete detector, data loader

- Integrate Scenic trainer base classes
- Implement training step with loss computation
- Add SGD optimizer with learning rate schedule
- Implement checkpointing with orbax
- Add logging and progress tracking
- Run initial training on CMR dataset (overfitting test)

## Known Issues

### Critical
- None currently

### High
- Type annotations incomplete in some modules (need jaxtyping for all array shapes)
- No CUDA/GPU testing yet (development on CPU)

### Medium
- FPN implementation needs verification against reference implementation
- RPN anchor generation may need tuning for different datasets
- Missing gradient checkpointing for memory optimization

### Low
- Documentation could use more code examples
- Need to add pre-commit hooks for code formatting
- Consider adding a changelog

## Future Enhancements (Backlog)

### Model Architecture
- [ ] Cascade R-CNN variant
- [ ] Deformable convolutions
- [ ] Feature pyramid attention
- [ ] Dynamic head for better mask quality

### Training & Optimization
- [ ] Knowledge distillation from larger models
- [ ] Self-supervised pretraining
- [ ] AutoAugment / RandAugment policies
- [ ] Test-time augmentation

### Deployment
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Model quantization (INT8)
- [ ] Edge deployment (Coral TPU, Jetson)
- [ ] Web deployment (WebGPU via JAX)

### Datasets & Benchmarks
- [ ] LVIS dataset support
- [ ] Open Images dataset support
- [ ] Custom dataset templates
- [ ] Benchmark suite across multiple datasets

### Developer Experience
- [ ] Interactive demo with Gradio
- [ ] Colab notebooks
- [ ] Docker containers
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Auto-generated API docs with Sphinx

## Contributing

We welcome contributions! Priority areas for community help:

1. **Testing**: Write unit tests for existing components
2. **Documentation**: Add docstrings, tutorials, examples
3. **Benchmarking**: Run and report benchmarks on different hardware
4. **Data Augmentation**: Implement advanced augmentation strategies
5. **Visualization**: Tools for visualizing predictions and errors

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coming soon).

## Milestones

### v0.1.0 - Minimum Viable Product (Target: TBD)
- Complete Mask R-CNN implementation
- Basic training loop
- COCO evaluation
- CMR dataset training working
- Documentation complete

### v0.2.0 - Production Ready (Target: TBD)
- Multi-GPU training
- Full Scenic integration
- COCO benchmark results
- Comprehensive testing (>80% coverage)
- API documentation

### v1.0.0 - Stable Release (Target: TBD)
- Stable API
- Multiple backbone support
- Competitive COCO benchmarks
- Deployment examples
- Tutorial notebooks

## Questions & Discussions

For questions, ideas, or discussions, please:
- Open an issue on GitHub
- Start a discussion in GitHub Discussions (when enabled)
- Reach out to maintainers

---

**Last Updated**: 2025-10-19
**Maintainers**: TBD
