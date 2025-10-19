# Dataset Configuration

This guide covers dataset setup and configuration for Detectax.

## COCO Format

Detectax uses the COCO JSON annotation format for training and evaluation. Your dataset should follow this structure:

```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── train.json
    └── val.json
```

### COCO JSON Format

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "height": 480, "width": 640}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "person"}
  ]
}
```

## CMR Dataset (Development/Testing)

For development and testing, the project uses the CMR dataset:

### File Paths

- **Training annotations**: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
- **Validation annotations**: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`
- **Images**: `/home/georgepearse/data/images`

### Configuration Example

```yaml
# configs/cmr_config.yaml
data:
  dataset: "cmr"
  train_annotation: "/path/to/cmr/annotations/train_ordered.json"
  val_annotation: "/path/to/cmr/annotations/val_ordered.json"
  image_dir: "/path/to/images"
  num_classes: null  # Auto-detect from annotations
```

### Class Count Verification

**Important**: Always verify that the number of classes in your annotations matches your model configuration:

```python
import json

with open('/path/to/annotations.json', 'r') as f:
    data = json.load(f)

num_classes = len(data['categories'])
print(f"Number of classes: {num_classes}")
```

Then ensure your model config matches:

```yaml
model:
  num_classes: <num_classes>  # Must match annotation file
```

## Standard COCO Dataset

To use the standard COCO 2017 dataset:

### Download

```bash
# Download COCO 2017
mkdir -p data/coco
cd data/coco

# Images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Directory Structure

```
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Configuration

```yaml
# configs/coco_config.yaml
data:
  dataset: "coco"
  train_annotation: "data/coco/annotations/instances_train2017.json"
  val_annotation: "data/coco/annotations/instances_val2017.json"
  train_image_dir: "data/coco/train2017"
  val_image_dir: "data/coco/val2017"
  num_classes: 80
```

## Custom Dataset

To use your own dataset:

1. **Format annotations** in COCO JSON format (see above)
2. **Organize images** in a single directory or split by train/val
3. **Create config file**:

```yaml
# configs/custom_config.yaml
data:
  dataset: "custom"
  train_annotation: "/path/to/your/train.json"
  val_annotation: "/path/to/your/val.json"
  image_dir: "/path/to/your/images"
  num_classes: <your_num_classes>
```

4. **Validate your annotations**:

```python
from pycocotools.coco import COCO

# This will raise errors if format is invalid
coco = COCO('/path/to/your/annotations.json')
print(f"Loaded {len(coco.imgs)} images")
print(f"Loaded {len(coco.anns)} annotations")
print(f"Categories: {[cat['name'] for cat in coco.loadCats(coco.getCatIds())]}")
```

## Data Augmentation

Detectax supports common data augmentations for object detection:

```yaml
data:
  augmentation:
    random_flip: true
    random_crop: true
    color_jitter: true
    mosaic: false  # Advanced augmentation
    mixup: false   # Advanced augmentation
```

See the training configuration documentation for more augmentation options.
