"""Mask R-CNN detector module implemented with Flax.

This module combines the backbone, FPN neck, RPN proposal network, a simple
Fast R-CNN style detection head, and an FCN-style mask head to form a complete
Mask R-CNN style detector that supports both training (loss computation) and
inference (prediction) flows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from detectax.models.backbones.resnet import ResNetBackbone
from detectax.models.heads.rpn import RPN
from detectax.models.layers.roi_align import roi_align
from detectax.models.losses.detection_loss import detection_loss
from detectax.models.losses.mask_loss import mask_loss
from detectax.models.losses.rpn_loss import rpn_loss
from detectax.models.necks.fpn import FPN
from detectax.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from detectax.models.task_modules.assigners.detection_assigner import assign_detection_targets
from detectax.models.task_modules.assigners.rpn_assigner import assign_rpn_targets
from detectax.models.task_modules.mask_target_generator import generate_mask_targets
from detectax.models.utils.anchor_generator import AnchorGenerator
from detectax.models.utils.box_coder import decode_boxes
from detectax.models.utils.nms import nms

# ---------------------------------------------------------------------------
# Type aliases for readability.
# ---------------------------------------------------------------------------
Images = Float[Array, "batch height width channels"]
RPNLevelScores = Float[Array, "batch height width anchors"]
RPNLevelDeltas = Float[Array, "batch height width anchor_deltas"]
ProposalBoxes = Float[Array, "batch num_boxes 4"]
DetectionScores = Float[Array, "batch num_boxes"]
DetectionLabels = Int[Array, "batch num_boxes"]
MaskOutputs = Float[Array, "batch num_boxes mask_h mask_w"]


@dataclass
class MaskRCNNTargets:
    """Container for ground-truth annotations used when training."""

    boxes: Float[Array, "batch num_targets 4"]
    labels: Int[Array, "batch num_targets"]
    masks: Float[Array, "batch num_targets mask_height mask_width"]


@dataclass
class MaskHeadTargets:
    """Container for mask supervision."""

    masks: Float[Array, "batch num_rois mask_height mask_width"]
    classes: Int[Array, "batch num_rois"]


@dataclass
class AssignedTargets:
    """Structured bundle of per-stage training targets."""

    rpn_objectness: Int[Array, "batch num_anchors"]
    rpn_deltas: Float[Array, "batch num_anchors 4"]
    rpn_weights: Float[Array, "batch num_anchors"]
    detection_labels: Int[Array, "batch num_rois"]
    detection_deltas: Float[Array, "batch num_rois num_classes 4"]
    detection_weights: Float[Array, "batch num_rois"]
    mask_targets: MaskHeadTargets
    mask_positive: Bool[Array, "batch num_rois"]


@dataclass
class MaskRCNNConfig:
    """Configuration bundle for the Mask R-CNN detector."""

    num_classes: int = 1
    num_proposals: int = 100
    score_threshold: float = 0.05
    class_agnostic_bbox: bool = True
    roi_pool_size: int = 7
    mask_pool_size: int = 14
    roi_pooling_level: str = "P2"
    mask_pooling_level: str = "P2"
    backbone: dict[str, Any] = field(default_factory=dict)
    fpn: dict[str, Any] = field(default_factory=dict)
    rpn: dict[str, Any] = field(default_factory=dict)
    detection_head: dict[str, Any] = field(default_factory=dict)
    mask_head: dict[str, Any] = field(default_factory=dict)
    anchor_generator: dict[str, Any] = field(default_factory=dict)


class DetectionHead(nn.Module):
    """Simple Fast R-CNN style detection head with two fully-connected layers."""

    num_classes: int
    hidden_dim: int = 1024
    class_agnostic_bbox: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        roi_features: Float[Array, "batch num_rois height width channels"],
        *,
        train: bool = False,
    ) -> tuple[
        Float[Array, "batch num_rois num_classes_plus_bg"],
        Float[Array, "batch num_rois bbox_params"],
    ]:
        del train  # The head is identical for training and inference.

        if roi_features.ndim != 5:
            raise ValueError(f"Expected RoI features with 5 dimensions, got shape {roi_features.shape}.")

        batch, num_rois, height, width, channels = roi_features.shape
        flattened = roi_features.reshape((batch * num_rois, height * width * channels))

        hidden = nn.Dense(self.hidden_dim, dtype=self.dtype, name="fc1")(flattened)
        hidden = nn.relu(hidden)
        hidden = nn.Dense(self.hidden_dim, dtype=self.dtype, name="fc2")(hidden)
        hidden = nn.relu(hidden)

        cls_logits = nn.Dense(self.num_classes + 1, dtype=self.dtype, name="cls_logits")(hidden)
        bbox_dim = 4 if self.class_agnostic_bbox else self.num_classes * 4
        bbox_deltas = nn.Dense(bbox_dim, dtype=self.dtype, name="bbox_deltas")(hidden)

        cls_logits = cls_logits.reshape((batch, num_rois, self.num_classes + 1))
        bbox_deltas = bbox_deltas.reshape((batch, num_rois, bbox_dim))
        return cls_logits, bbox_deltas


class MaskRCNN(nn.Module):
    """Mask R-CNN detector integrating backbone, FPN, RPN, and ROI heads."""

    config: MaskRCNNConfig

    def setup(self) -> None:
        """Instantiate sub-modules based on the provided configuration."""
        cfg = self.config

        backbone_cfg = {"num_layers": 50, "num_filters": 64, "dtype": jnp.float32}
        backbone_cfg.update(cfg.backbone)
        self.backbone = ResNetBackbone(**backbone_cfg)

        fpn_cfg = {
            "in_channels": [256, 512, 1024, 2048],
            "out_channels": 256,
            "num_outs": 5,
            "add_extra_convs": "on_input",
        }
        fpn_cfg.update(cfg.fpn)
        self.fpn = FPN(**fpn_cfg)

        rpn_cfg = {"in_channels": fpn_cfg["out_channels"], "feat_channels": 256, "num_anchors": 3}
        rpn_cfg.update(cfg.rpn)
        self.rpn = RPN(**rpn_cfg)

        detection_head_cfg = {"hidden_dim": 1024, "class_agnostic_bbox": cfg.class_agnostic_bbox}
        detection_head_cfg.update(cfg.detection_head)
        self.detection_head = DetectionHead(num_classes=cfg.num_classes, **detection_head_cfg)

        mask_head_cfg = {"num_convs": 4, "conv_features": 256}
        mask_head_cfg.update(cfg.mask_head)
        self.mask_head = FCNMaskHead(num_classes=cfg.num_classes, **mask_head_cfg)

        # Configure anchor generator to match the number of FPN outputs.
        default_levels = ("P2", "P3", "P4", "P5", "P6")
        num_levels = fpn_cfg["num_outs"]
        level_names = default_levels[:num_levels]

        default_anchor = AnchorGenerator()
        strides = tuple(default_anchor.strides[:num_levels])
        base_sizes = tuple(default_anchor.base_sizes[:num_levels])
        anchor_cfg = {
            "strides": strides,
            "base_sizes": base_sizes,
            "aspect_ratios": default_anchor.aspect_ratios,
            "scales": default_anchor.scales,
            "level_names": level_names,
        }
        anchor_cfg.update(cfg.anchor_generator)
        self.anchor_generator = AnchorGenerator(**anchor_cfg)

        self.level_names = anchor_cfg["level_names"]
        self.backbone_stage_order = ("stage_1", "stage_2", "stage_3", "stage_4")

    def __call__(
        self,
        images: Images,
        *,
        training: bool = False,
        targets: MaskRCNNTargets | None = None,
    ) -> dict[str, Float[Array, ""]] | list[dict[str, Float[Array, ...]]]:
        """Run the full detector in either training or inference mode."""
        image_height = images.shape[1]
        image_width = images.shape[2]
        image_shape = (image_height, image_width)

        backbone_features = self.backbone(images, train=training)
        pyramid_inputs = [backbone_features[name] for name in self.backbone_stage_order]
        fpn_features = self.fpn(pyramid_inputs, train=training)

        # Convert FPN outputs into both list and dict representations.
        fpn_list = list(fpn_features)
        fpn_dict = {level: feat for level, feat in zip(self.level_names, fpn_list)}

        feature_shapes = [(feat.shape[1], feat.shape[2]) for feat in fpn_list]
        anchors_per_level = self.anchor_generator.generate(feature_map_shapes=feature_shapes, per_level=True)

        rpn_cls_scores, rpn_bbox_deltas = self.rpn(fpn_list, train=training)
        proposals, proposal_objectness = self._generate_proposals(
            rpn_cls_scores,
            rpn_bbox_deltas,
            fpn_list,
            image_shape,
            anchors_per_level=anchors_per_level,
        )

        roi_level = self.config.roi_pooling_level
        if roi_level not in fpn_dict:
            raise KeyError(f"Requested RoI pooling level '{roi_level}' not found in FPN outputs {tuple(fpn_dict.keys())}.")
        roi_features = self._roi_align_per_image(
            fpn_dict[roi_level],
            proposals,
            output_size=(self.config.roi_pool_size, self.config.roi_pool_size),
            image_shape=image_shape,
        )

        mask_level = self.config.mask_pooling_level
        mask_level = mask_level if mask_level in fpn_dict else roi_level
        mask_features = self._roi_align_per_image(
            fpn_dict[mask_level],
            proposals,
            output_size=(self.config.mask_pool_size, self.config.mask_pool_size),
            image_shape=image_shape,
        )

        cls_logits, bbox_deltas = self.detection_head(roi_features, train=training)

        if self.config.class_agnostic_bbox:
            bbox_deltas = bbox_deltas
        else:
            bbox_deltas = bbox_deltas  # Placeholder for future class-specific handling.

        refined_boxes = decode_boxes(bbox_deltas, proposals)
        refined_boxes = self._clip_boxes(refined_boxes, image_shape)

        mask_logits = self.mask_head(mask_features, train=training)

        if training:
            if targets is None:
                raise ValueError("Mask R-CNN training requires targets, but none were provided.")
            assigned = self._assign_targets(
                targets,
                proposals,
                anchors_per_level,
                mask_shape=(mask_logits.shape[2], mask_logits.shape[3]),
            )
            return self._compute_losses(
                rpn_cls_scores,
                rpn_bbox_deltas,
                cls_logits,
                bbox_deltas,
                mask_logits,
                assigned,
            )

        predictions = self._format_predictions(
            refined_boxes,
            cls_logits,
            proposal_objectness,
            mask_logits,
        )
        return predictions

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _generate_proposals(
        self,
        cls_scores_list: Sequence[RPNLevelScores],
        bbox_deltas_list: Sequence[RPNLevelDeltas],
        fpn_features: Sequence[Float[Array, "batch height width channels"]],
        image_shape: tuple[int, int],
        *,
        anchors_per_level: dict[str, Float[Array, "num_anchors 4"]] | None = None,
    ) -> tuple[ProposalBoxes, DetectionScores]:
        """Decode RPN outputs into proposal boxes and select top candidates."""
        if anchors_per_level is None:
            feature_shapes = [(feat.shape[1], feat.shape[2]) for feat in fpn_features]
            anchors_per_level = self.anchor_generator.generate(feature_map_shapes=feature_shapes, per_level=True)

        boxes_all: list[Float[Array, "batch num_level_anchors 4"]] = []
        scores_all: list[Float[Array, "batch num_level_anchors"]] = []
        for level_name, cls_scores, bbox_deltas in zip(self.level_names, cls_scores_list, bbox_deltas_list):
            anchors = anchors_per_level[level_name]
            batch_size, height, width, num_anchors = cls_scores.shape

            cls_probs = jax.nn.sigmoid(cls_scores).reshape((batch_size, -1))
            bbox_deltas = bbox_deltas.reshape((batch_size, height, width, num_anchors, 4))
            bbox_deltas = bbox_deltas.reshape((batch_size, -1, 4))

            anchors = jnp.broadcast_to(anchors[None, ...], (batch_size, anchors.shape[0], 4))
            decoded = decode_boxes(bbox_deltas, anchors)
            decoded = self._clip_boxes(decoded, image_shape)

            boxes_all.append(decoded)
            scores_all.append(cls_probs)

        if boxes_all:
            all_boxes = jnp.concatenate(boxes_all, axis=1)
            all_scores = jnp.concatenate(scores_all, axis=1)
        else:
            batch_size = fpn_features[0].shape[0] if fpn_features else 1
            empty_boxes = jnp.zeros((batch_size, 0, 4), dtype=jnp.float32)
            empty_scores = jnp.zeros((batch_size, 0), dtype=jnp.float32)
            return empty_boxes, empty_scores

        num_candidates = all_boxes.shape[1]
        max_keep = min(self.config.num_proposals, num_candidates)

        if max_keep == 0:
            batch_size = all_boxes.shape[0]
            empty_boxes = jnp.zeros((batch_size, 0, 4), dtype=all_boxes.dtype)
            empty_scores = jnp.zeros((batch_size, 0), dtype=all_scores.dtype)
            return empty_boxes, empty_scores

        nms_result = nms(all_boxes, all_scores, iou_threshold=0.7, max_output_size=max_keep)
        indices = nms_result.indices
        valid_mask = indices >= 0
        safe_indices = jnp.where(valid_mask, indices, 0)

        top_boxes = jnp.take_along_axis(all_boxes, safe_indices[..., None], axis=1)
        top_scores = jnp.take_along_axis(all_scores, safe_indices, axis=1)

        top_boxes = jnp.where(valid_mask[..., None], top_boxes, 0.0)
        top_scores = jnp.where(valid_mask, top_scores, 0.0)
        return top_boxes, top_scores

    def _roi_align_per_image(
        self,
        feature_map: Float[Array, "batch height width channels"],
        boxes: ProposalBoxes,
        *,
        output_size: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> Float[Array, "batch num_boxes out_h out_w channels"]:
        """Apply RoI Align independently for each image in the batch."""
        batch_size = feature_map.shape[0]
        num_boxes = boxes.shape[1]
        out_h, out_w = output_size
        spatial_scale = feature_map.shape[1] / float(image_shape[0])

        aligned_outputs: list[Float[Array, "num_boxes out_h out_w channels"]] = []
        for i in range(batch_size):
            image_boxes = boxes[i]
            single_feature = feature_map[i : i + 1]
            if num_boxes == 0:
                aligned = jnp.zeros((0, out_h, out_w, feature_map.shape[-1]), dtype=feature_map.dtype)
            else:
                aligned = roi_align(
                    single_feature,
                    image_boxes,
                    output_size=output_size,
                    spatial_scale=spatial_scale,
                )
            aligned_outputs.append(aligned)

        stacked = jnp.stack(aligned_outputs, axis=0)
        return stacked

    def _clip_boxes(
        self,
        boxes: ProposalBoxes,
        image_shape: tuple[int, int],
    ) -> ProposalBoxes:
        """Clamp box coordinates to the valid image region."""
        height, width = image_shape
        x1 = jnp.clip(boxes[..., 0], a_min=0.0, a_max=width - 1.0)
        y1 = jnp.clip(boxes[..., 1], a_min=0.0, a_max=height - 1.0)
        x2 = jnp.clip(boxes[..., 2], a_min=0.0, a_max=width - 1.0)
        y2 = jnp.clip(boxes[..., 3], a_min=0.0, a_max=height - 1.0)
        return jnp.stack((x1, y1, x2, y2), axis=-1)

    def _assign_targets(
        self,
        targets: MaskRCNNTargets,
        proposals: ProposalBoxes,
        anchors_per_level: dict[str, Float[Array, "num_anchors 4"]],
        *,
        mask_shape: tuple[int, int],
    ) -> AssignedTargets:
        """Assign ground-truth annotations to anchors, proposals, and masks."""
        mask_height, mask_width = mask_shape
        if mask_height != mask_width:
            raise ValueError(f"Mask supervision requires square targets; received {mask_shape}.")

        anchor_sequences = [anchors_per_level[level] for level in self.level_names if level in anchors_per_level]
        anchors_concat = jnp.concatenate(anchor_sequences, axis=0) if anchor_sequences else jnp.zeros((0, 4), dtype=jnp.float32)

        batch_size, num_rois = proposals.shape[:2]
        num_classes = self.config.num_classes

        rpn_objectness_list: list[jnp.ndarray] = []
        rpn_deltas_list: list[jnp.ndarray] = []
        rpn_weights_list: list[jnp.ndarray] = []
        detection_labels_list: list[jnp.ndarray] = []
        detection_deltas_list: list[jnp.ndarray] = []
        detection_weights_list: list[jnp.ndarray] = []

        mask_targets_np = np.zeros((batch_size, num_rois, mask_height, mask_width), dtype=np.float32)
        mask_positive_np = np.zeros((batch_size, num_rois), dtype=bool)

        has_masks = targets.masks is not None

        for i in range(batch_size):
            gt_boxes_np = np.asarray(targets.boxes[i])
            gt_labels_np = np.asarray(targets.labels[i])
            masks_per_image = targets.masks[i] if has_masks else ()

            valid_indices = np.where(gt_labels_np > 0)[0]
            if valid_indices.size > 0:
                boxes_valid = jnp.asarray(gt_boxes_np[valid_indices], dtype=jnp.float32)
                labels_valid = jnp.asarray(gt_labels_np[valid_indices], dtype=jnp.int32)
                if has_masks:
                    if isinstance(masks_per_image, (list, tuple)):
                        masks_valid = [masks_per_image[int(idx)] for idx in valid_indices]
                    else:
                        masks_np = np.asarray(masks_per_image)
                        masks_valid = [masks_np[int(idx)] for idx in valid_indices]
                else:
                    masks_valid = []
            else:
                boxes_valid = jnp.zeros((0, 4), dtype=jnp.float32)
                labels_valid = jnp.zeros((0,), dtype=jnp.int32)
                masks_valid = []

            rpn_labels_i, rpn_deltas_i, rpn_weights_i = assign_rpn_targets(anchors_concat, boxes_valid, labels_valid)
            rpn_objectness_list.append(rpn_labels_i.astype(jnp.int32))
            rpn_deltas_list.append(rpn_deltas_i.astype(jnp.float32))
            rpn_weights_list.append(rpn_weights_i.astype(jnp.float32))

            proposals_i = proposals[i]
            if boxes_valid.shape[0] > 0:
                det_labels_i, det_deltas_full, _ = assign_detection_targets(
                    proposals_i,
                    boxes_valid,
                    labels_valid,
                )
            else:
                det_labels_i = jnp.zeros((num_rois,), dtype=jnp.int32)
                if num_classes > 0:
                    det_deltas_full = jnp.zeros((num_rois, num_classes + 1, 4), dtype=jnp.float32)
                else:
                    det_deltas_full = jnp.zeros((num_rois, 1, 4), dtype=jnp.float32)

            if num_classes > 0:
                if det_deltas_full.shape[1] == num_classes + 1:
                    det_deltas = det_deltas_full[:, 1:, :]
                elif det_deltas_full.shape[1] == num_classes:
                    det_deltas = det_deltas_full
                else:
                    raise ValueError(
                        f"Detection target class dimension {det_deltas_full.shape[1]} does not match configured num_classes {num_classes}."
                    )
            else:
                det_deltas = jnp.zeros((num_rois, 0, 4), dtype=jnp.float32)

            det_labels_i = jnp.clip(det_labels_i, 0, num_classes)
            detection_labels_list.append(det_labels_i.astype(jnp.int32))
            detection_deltas_list.append(det_deltas.astype(jnp.float32))

            roi_weights_i = jnp.where(det_labels_i >= 0, 1.0, 0.0).astype(jnp.float32)
            detection_weights_list.append(roi_weights_i)

            positive_mask = np.asarray(det_labels_i > 0, dtype=bool)
            mask_positive_np[i] = positive_mask
            if positive_mask.any() and masks_valid:
                pos_indices = np.where(positive_mask)[0]
                mask_size = mask_height
                mask_targets_pos = generate_mask_targets(
                    np.asarray(proposals_i)[pos_indices],
                    np.asarray(boxes_valid),
                    masks_valid,
                    mask_size=mask_size,
                )
                mask_targets_np[i, pos_indices, :, :] = np.asarray(mask_targets_pos)

        rpn_objectness = jnp.stack(rpn_objectness_list, axis=0) if rpn_objectness_list else jnp.zeros((0, 0), dtype=jnp.int32)
        rpn_deltas = jnp.stack(rpn_deltas_list, axis=0) if rpn_deltas_list else jnp.zeros((0, 0, 4), dtype=jnp.float32)
        rpn_weights = jnp.stack(rpn_weights_list, axis=0) if rpn_weights_list else jnp.zeros((0, 0), dtype=jnp.float32)

        detection_labels = jnp.stack(detection_labels_list, axis=0) if detection_labels_list else jnp.zeros((0, 0), dtype=jnp.int32)
        detection_deltas = (
            jnp.stack(detection_deltas_list, axis=0)
            if detection_deltas_list
            else jnp.zeros((0, 0, max(self.config.num_classes, 0), 4), dtype=jnp.float32)
        )
        detection_weights = jnp.stack(detection_weights_list, axis=0) if detection_weights_list else jnp.zeros((0, 0), dtype=jnp.float32)

        mask_targets = MaskHeadTargets(
            masks=jnp.asarray(mask_targets_np, dtype=jnp.float32),
            classes=detection_labels,
        )
        mask_positive = jnp.asarray(mask_positive_np, dtype=jnp.bool_)

        return AssignedTargets(
            rpn_objectness=rpn_objectness,
            rpn_deltas=rpn_deltas,
            rpn_weights=rpn_weights,
            detection_labels=detection_labels,
            detection_deltas=detection_deltas,
            detection_weights=detection_weights,
            mask_targets=mask_targets,
            mask_positive=mask_positive,
        )

    def _compute_losses(
        self,
        rpn_cls_scores: Sequence[RPNLevelScores],
        rpn_bbox_deltas: Sequence[RPNLevelDeltas],
        cls_logits: Float[Array, "batch num_rois num_classes_plus_bg"],
        bbox_deltas: Float[Array, "batch num_rois bbox_params"],
        mask_logits: Float[Array, "batch num_rois mask_h mask_w num_classes"],
        assigned: AssignedTargets,
    ) -> dict[str, Float[Array, ""]]:
        """Compute supervised losses for Mask R-CNN training."""
        batch_size = cls_logits.shape[0]

        objectness_levels = [level.reshape((batch_size, -1)) for level in rpn_cls_scores]
        objectness_pred = jnp.concatenate(objectness_levels, axis=1) if objectness_levels else jnp.zeros((batch_size, 0), dtype=jnp.float32)

        bbox_levels: list[jnp.ndarray] = []
        for level in rpn_bbox_deltas:
            b, h, w, channels = level.shape
            if channels == 0:
                bbox_levels.append(jnp.zeros((b, 0, 4), dtype=level.dtype))
                continue
            num_anchors = channels // 4
            reshaped = level.reshape((b, h, w, num_anchors, 4)).reshape((b, -1, 4))
            bbox_levels.append(reshaped)
        bbox_pred = jnp.concatenate(bbox_levels, axis=1) if bbox_levels else jnp.zeros((batch_size, 0, 4), dtype=jnp.float32)

        _, rpn_cls_loss, rpn_reg_loss = rpn_loss(
            objectness_pred,
            bbox_pred,
            assigned.rpn_objectness,
            assigned.rpn_deltas,
            assigned.rpn_weights,
        )

        if self.config.class_agnostic_bbox and self.config.num_classes > 0:
            bbox_deltas_for_loss = jnp.repeat(bbox_deltas[..., None, :], self.config.num_classes, axis=2)
        else:
            bbox_deltas_for_loss = bbox_deltas

        _, det_cls_loss, det_reg_loss = detection_loss(
            cls_logits,
            bbox_deltas_for_loss,
            assigned.detection_labels,
            assigned.detection_deltas,
            assigned.detection_weights,
        )

        mask_logits_for_loss = jnp.transpose(mask_logits, (0, 1, 4, 2, 3))
        mask_loss_value = mask_loss(mask_logits_for_loss, assigned.mask_targets, assigned.mask_positive)

        total = rpn_cls_loss + rpn_reg_loss + det_cls_loss + det_reg_loss + mask_loss_value

        return {
            "rpn_cls": rpn_cls_loss,
            "rpn_reg": rpn_reg_loss,
            "det_cls": det_cls_loss,
            "det_reg": det_reg_loss,
            "mask": mask_loss_value,
            "total": total,
        }

    def _format_predictions(
        self,
        boxes: ProposalBoxes,
        cls_logits: Float[Array, "batch num_rois num_classes_plus_bg"],
        objectness_scores: DetectionScores,
        mask_logits: Float[Array, "batch num_rois mask_h mask_w num_classes"],
    ) -> list[dict[str, Float[Array, ...]]]:
        """Convert raw network outputs into user-friendly prediction dicts."""
        class_probs = jax.nn.softmax(cls_logits, axis=-1)
        mask_probs = jax.nn.sigmoid(mask_logits)

        if self.config.num_classes == 0:
            zero_scores = jnp.zeros_like(objectness_scores)
            zero_labels = jnp.zeros_like(objectness_scores, dtype=jnp.int32)
            zero_masks = jnp.zeros(
                (boxes.shape[0], boxes.shape[1], mask_probs.shape[2], mask_probs.shape[3]),
                dtype=mask_probs.dtype,
            )

            predictions = []
            for i in range(boxes.shape[0]):
                predictions.append(
                    {
                        "boxes": boxes[i],
                        "scores": zero_scores[i],
                        "labels": zero_labels[i],
                        "masks": zero_masks[i],
                    }
                )
            return predictions

        fg_probs = class_probs[..., 1:]

        batch_size = boxes.shape[0]
        per_image_max = boxes.shape[1]
        mask_height = mask_probs.shape[2]
        mask_width = mask_probs.shape[3]

        predictions: list[dict[str, Float[Array, ...]]] = []

        for i in range(batch_size):
            image_boxes = boxes[i]
            image_fg_probs = fg_probs[i]
            image_objectness = objectness_scores[i]
            image_masks = mask_probs[i]

            if per_image_max == 0:
                predictions.append(
                    {
                        "boxes": image_boxes,
                        "scores": jnp.zeros((0,), dtype=image_objectness.dtype),
                        "labels": jnp.zeros((0,), dtype=jnp.int32),
                        "masks": jnp.zeros((0, mask_height, mask_width), dtype=image_masks.dtype),
                    }
                )
                continue

            per_class_boxes = []
            per_class_scores = []
            per_class_labels = []
            per_class_masks = []

            for class_idx in range(self.config.num_classes):
                class_scores = image_fg_probs[:, class_idx] * image_objectness
                nms_result = nms(image_boxes, class_scores, iou_threshold=0.5, max_output_size=per_image_max)
                class_indices = nms_result.indices
                valid_mask = class_indices >= 0
                safe_indices = jnp.where(valid_mask, class_indices, 0)

                selected_boxes = jnp.take(image_boxes, safe_indices, axis=0)
                selected_scores = jnp.take(class_scores, safe_indices, axis=0)
                selected_masks = jnp.take(image_masks[..., class_idx], safe_indices, axis=0)

                selected_boxes = jnp.where(valid_mask[:, None], selected_boxes, 0.0)
                selected_scores = jnp.where(valid_mask, selected_scores, 0.0)
                label_values = jnp.full((per_image_max,), class_idx + 1, dtype=jnp.int32)
                selected_labels = jnp.where(valid_mask, label_values, 0)
                selected_masks = jnp.where(valid_mask[:, None, None], selected_masks, 0.0)

                per_class_boxes.append(selected_boxes)
                per_class_scores.append(selected_scores)
                per_class_labels.append(selected_labels)
                per_class_masks.append(selected_masks)

            boxes_stack = jnp.stack(per_class_boxes, axis=0)
            scores_stack = jnp.stack(per_class_scores, axis=0)
            labels_stack = jnp.stack(per_class_labels, axis=0)
            masks_stack = jnp.stack(per_class_masks, axis=0)

            flat_boxes = boxes_stack.reshape((-1, 4))
            flat_scores = scores_stack.reshape((-1,))
            flat_labels = labels_stack.reshape((-1,))
            flat_masks = masks_stack.reshape((-1, mask_height, mask_width))

            top_k = min(per_image_max, flat_scores.shape[0])
            if top_k == 0:
                predictions.append(
                    {
                        "boxes": jnp.zeros((0, 4), dtype=image_boxes.dtype),
                        "scores": jnp.zeros((0,), dtype=image_objectness.dtype),
                        "labels": jnp.zeros((0,), dtype=jnp.int32),
                        "masks": jnp.zeros((0, mask_height, mask_width), dtype=image_masks.dtype),
                    }
                )
                continue

            top_scores, top_indices = jax.lax.top_k(flat_scores, k=top_k)
            top_boxes = jnp.take(flat_boxes, top_indices, axis=0)
            top_labels = jnp.take(flat_labels, top_indices, axis=0)
            top_masks = jnp.take(flat_masks, top_indices, axis=0)

            valid_top = top_scores > self.config.score_threshold
            final_scores = jnp.where(valid_top, top_scores, 0.0)
            final_labels = jnp.where(valid_top, top_labels, 0).astype(jnp.int32)
            final_boxes = jnp.where(valid_top[:, None], top_boxes, 0.0)
            final_masks = jnp.where(valid_top[:, None, None], top_masks, 0.0)

            predictions.append(
                {
                    "boxes": final_boxes,
                    "scores": final_scores,
                    "labels": final_labels,
                    "masks": final_masks,
                }
            )

        return predictions


__all__ = ["MaskRCNN", "MaskRCNNConfig", "MaskRCNNTargets"]
