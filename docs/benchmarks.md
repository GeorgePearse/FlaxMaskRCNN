# Performance Benchmarks

This document contains performance benchmarks for Detectax across different configurations and datasets.

## Baseline Results

> **Note**: Detectax is currently in development. Benchmark results will be added as the implementation progresses.

### COCO 2017 Val (Expected)

Target performance metrics on COCO 2017 validation set:

| Backbone | mAP | AP50 | AP75 | APs | APm | APl | FPS (V100) |
|----------|-----|------|------|-----|-----|-----|------------|
| ResNet-50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ResNet-101 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Target benchmarks (based on reference implementations):**
- ResNet-50-FPN: ~37-38 mAP
- ResNet-101-FPN: ~39-40 mAP

### CMR Dataset

Performance on the CMR development dataset will be reported here.

| Configuration | mAP | AP50 | AP75 | Training Time | Notes |
|---------------|-----|------|------|---------------|-------|
| ResNet-50-FPN | TBD | TBD | TBD | TBD | Development config |

## Hardware Configurations

Benchmarks are measured on the following hardware:

### GPU Configurations
- **V100**: NVIDIA Tesla V100 32GB
- **A100**: NVIDIA A100 40GB (planned)
- **TPU v3**: Google TPU v3-8 (planned)

### CPU Baseline
- **System**: TBD
- **CPU**: TBD
- **RAM**: TBD

## Training Performance

### Training Speed

| Backbone | Batch Size | GPU | Images/sec | Hours/epoch (COCO) |
|----------|-----------|-----|------------|-------------------|
| ResNet-50 | 16 | V100 | TBD | TBD |
| ResNet-101 | 16 | V100 | TBD | TBD |

### Memory Usage

| Backbone | Batch Size | Peak Memory (GB) | Notes |
|----------|-----------|------------------|-------|
| ResNet-50 | 16 | TBD | Single GPU |
| ResNet-101 | 16 | TBD | Single GPU |

## Inference Performance

### Latency

| Backbone | Batch Size | GPU | Latency (ms) | Throughput (images/s) |
|----------|-----------|-----|--------------|----------------------|
| ResNet-50 | 1 | V100 | TBD | TBD |
| ResNet-50 | 8 | V100 | TBD | TBD |

### Model Size

| Backbone | Parameters | Model Size (FP32) | Model Size (FP16) |
|----------|-----------|-------------------|-------------------|
| ResNet-50 | TBD M | TBD MB | TBD MB |
| ResNet-101 | TBD M | TBD MB | TBD MB |

## Optimization Techniques

Planned optimizations and their expected impact:

- [ ] **Mixed Precision (FP16)**: ~2x speedup expected
- [ ] **XLA Optimization**: ~1.3-1.5x speedup
- [ ] **Multi-GPU Training**: Linear scaling expected
- [ ] **Gradient Checkpointing**: Reduce memory, ~20% slower
- [ ] **TensorFloat-32 (TF32)**: ~1.2x speedup on A100

## Reproducibility

### Environment

```yaml
# Environment for benchmark reproduction
python: 3.12
jax: 0.4.35
jaxlib: 0.4.35+cuda12
flax: 0.10.2
optax: 0.2.4
cuda: 12.x
cudnn: 8.x
```

### Training Configuration

```yaml
# Standard training config for benchmarks
training:
  optimizer: sgd
  learning_rate: 0.02
  momentum: 0.9
  weight_decay: 0.0001
  batch_size: 16
  num_epochs: 12
  warmup_epochs: 1
  lr_schedule: multistep  # decay at epoch 8, 11
```

### Evaluation Protocol

1. **Standard COCO Evaluation**: Use pycocotools for official metrics
2. **IoU Thresholds**: 0.5:0.05:0.95 (standard COCO)
3. **Max Detections**: 100 per image
4. **NMS Threshold**: 0.5

## Comparison with Other Frameworks

| Framework | Backbone | mAP | Training Speed | Inference Speed |
|-----------|----------|-----|----------------|-----------------|
| Detectax (JAX) | ResNet-50 | TBD | TBD | TBD |
| Detectron2 (PyTorch) | ResNet-50 | ~37.9 | Baseline | Baseline |
| MMDetection (PyTorch) | ResNet-50 | ~38.2 | ~1.0x | ~1.0x |
| TensorFlow OD API | ResNet-50 | ~37.5 | ~0.8x | ~0.9x |

## Future Benchmarks

Planned benchmark additions:

- [ ] Multi-GPU distributed training (2, 4, 8 GPUs)
- [ ] TPU training and inference
- [ ] Additional backbones (ViT, Swin Transformer)
- [ ] Quantization (INT8) performance
- [ ] ONNX export and inference
- [ ] Edge deployment (Coral, Jetson)

## Contributing Benchmarks

If you run Detectax benchmarks on your hardware, please contribute results via pull request!

Include:
- Hardware specs (GPU model, CPU, RAM)
- Software environment (JAX version, CUDA version)
- Configuration file used
- Full COCO evaluation results
- Training/inference speed measurements

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
