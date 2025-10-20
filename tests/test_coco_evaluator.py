import json

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from detectrax.evaluation import COCOEvaluator


def _write_dummy_annotations(tmp_path):
    annotations = {
        "images": [
            {
                "id": 1,
                "file_name": "dummy.jpg",
                "width": 100,
                "height": 100,
            }
        ],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "area": 2500,
                "iscrowd": 0,
                "segmentation": [[10, 10, 60, 10, 60, 60, 10, 60]],
            }
        ],
    }

    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotations))
    return annotation_path


def _dummy_predictions():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:60, 10:60] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")

    return {
        "bbox": [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 50, 50],
                "score": 0.9,
            }
        ],
        "segm": [
            {
                "image_id": 1,
                "category_id": 1,
                "segmentation": rle,
                "score": 0.9,
            }
        ],
    }


def test_coco_evaluator_perfect_prediction(tmp_path):
    annotation_path = _write_dummy_annotations(tmp_path)
    evaluator = COCOEvaluator(str(annotation_path))

    predictions = _dummy_predictions()
    metrics = evaluator.evaluate(predictions)

    bbox_metrics = metrics["bbox"]
    assert bbox_metrics.AP == pytest.approx(1.0, abs=1e-6)
    assert bbox_metrics.AP50 == pytest.approx(1.0, abs=1e-6)
    assert bbox_metrics.AP75 == pytest.approx(1.0, abs=1e-6)
    assert bbox_metrics.APm == pytest.approx(1.0, abs=1e-6)
    assert bbox_metrics.APs == -1.0  # small objects absent
    assert bbox_metrics.APl == -1.0  # large objects absent

    segm_metrics = metrics["segm"]
    assert segm_metrics.AP == pytest.approx(1.0, abs=1e-6)
    assert segm_metrics.AP50 == pytest.approx(1.0, abs=1e-6)
    assert segm_metrics.AP75 == pytest.approx(1.0, abs=1e-6)
    assert segm_metrics.APm == pytest.approx(1.0, abs=1e-6)
    assert segm_metrics.APs == -1.0
    assert segm_metrics.APl == -1.0


def test_coco_evaluator_requires_predictions_per_iou_type(tmp_path):
    annotation_path = _write_dummy_annotations(tmp_path)
    evaluator = COCOEvaluator(str(annotation_path), iou_types=("bbox",))

    with pytest.raises(KeyError):
        evaluator.evaluate({"segm": []})
