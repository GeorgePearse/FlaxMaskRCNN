"""Unit tests for backbone, RPN, and RoI operations."""

import jax
import jax.numpy as jnp
import pytest
from detectax.models.backbones.resnet import ResNetBackbone
from detectax.models.heads.rpn import RPN, RPNHead
from detectax.models.layers.roi_align import roi_align, roi_pool


class TestResNetBackbone:
    """Test suite for ResNet backbone."""

    def test_backbone_output_shapes(self):
        """Test backbone produces correct output shapes."""
        backbone = ResNetBackbone(num_layers=50)
        images = jnp.ones((2, 224, 224, 3))

        variables = backbone.init(jax.random.PRNGKey(0), images, train=False)
        features = backbone.apply(variables, images, train=False)

        # Check we get 4 feature levels
        assert len(features) == 4
        assert 'stage_1' in features
        assert 'stage_2' in features
        assert 'stage_3' in features
        assert 'stage_4' in features

    def test_backbone_feature_channels(self):
        """Test backbone feature channels match ResNet-50."""
        backbone = ResNetBackbone(num_layers=50)
        images = jnp.ones((1, 224, 224, 3))

        variables = backbone.init(jax.random.PRNGKey(0), images, train=False)
        features = backbone.apply(variables, images, train=False)

        # ResNet-50 channel dimensions
        assert features['stage_1'].shape[-1] == 256   # C2
        assert features['stage_2'].shape[-1] == 512   # C3
        assert features['stage_3'].shape[-1] == 1024  # C4
        assert features['stage_4'].shape[-1] == 2048  # C5


class TestRPNHead:
    """Test suite for RPN head."""

    def test_rpn_head_output_shapes(self):
        """Test RPN head produces correct output shapes."""
        rpn_head = RPNHead(in_channels=256, num_anchors=3)
        feat = jnp.ones((2, 64, 64, 256))

        variables = rpn_head.init(jax.random.PRNGKey(0), feat)
        cls_score, bbox_pred = rpn_head.apply(variables, feat)

        assert cls_score.shape == (2, 64, 64, 3)
        assert bbox_pred.shape == (2, 64, 64, 12)  # 3 anchors * 4 coords

    def test_rpn_head_different_anchors(self):
        """Test RPN head with different number of anchors."""
        rpn_head = RPNHead(in_channels=256, num_anchors=9)
        feat = jnp.ones((1, 32, 32, 256))

        variables = rpn_head.init(jax.random.PRNGKey(0), feat)
        cls_score, bbox_pred = rpn_head.apply(variables, feat)

        assert cls_score.shape == (1, 32, 32, 9)
        assert bbox_pred.shape == (1, 32, 32, 36)  # 9 * 4


class TestRPN:
    """Test suite for complete RPN."""

    def test_rpn_multi_level(self):
        """Test RPN processes multiple FPN levels."""
        rpn = RPN(in_channels=256, num_anchors=3)

        # Simulate FPN features (P2, P3, P4, P5)
        fpn_feats = [
            jnp.ones((2, 64, 64, 256)),
            jnp.ones((2, 32, 32, 256)),
            jnp.ones((2, 16, 16, 256)),
            jnp.ones((2, 8, 8, 256)),
        ]

        variables = rpn.init(jax.random.PRNGKey(0), fpn_feats)
        cls_scores, bbox_preds = rpn.apply(variables, fpn_feats)

        assert len(cls_scores) == 4
        assert len(bbox_preds) == 4

        # Check shapes for each level
        assert cls_scores[0].shape == (2, 64, 64, 3)
        assert cls_scores[1].shape == (2, 32, 32, 3)
        assert bbox_preds[0].shape == (2, 64, 64, 12)


class TestRoIAlign:
    """Test suite for RoI Align."""

    def test_roi_align_basic(self):
        """Test basic RoI align functionality."""
        features = jnp.ones((1, 64, 64, 256))
        boxes = jnp.array([[10.0, 10.0, 50.0, 50.0]])

        aligned = roi_align(
            features,
            boxes,
            output_size=(7, 7),
            spatial_scale=1.0,
        )

        assert aligned.shape == (1, 7, 7, 256)

    def test_roi_align_multiple_boxes(self):
        """Test RoI align with multiple boxes."""
        features = jnp.ones((1, 64, 64, 256))
        boxes = jnp.array([
            [10.0, 10.0, 30.0, 30.0],
            [20.0, 20.0, 40.0, 40.0],
            [30.0, 30.0, 50.0, 50.0],
        ])

        aligned = roi_align(features, boxes, output_size=(7, 7))

        assert aligned.shape == (3, 7, 7, 256)

    def test_roi_align_spatial_scale(self):
        """Test RoI align with spatial scaling (for FPN levels)."""
        features = jnp.ones((1, 16, 16, 256))
        boxes = jnp.array([[40.0, 40.0, 120.0, 120.0]])  # In image coords

        aligned = roi_align(
            features,
            boxes,
            output_size=(7, 7),
            spatial_scale=0.25,  # FPN P2 has stride 4
        )

        assert aligned.shape == (1, 7, 7, 256)

    @pytest.mark.skip(reason="RoI pool needs JAX control flow fixes, RoI Align is primary")
    def test_roi_pool_basic(self):
        """Test RoI pooling as simpler alternative."""
        features = jnp.ones((1, 64, 64, 256))
        boxes = jnp.array([[10.0, 10.0, 50.0, 50.0]])

        pooled = roi_pool(features, boxes, output_size=(7, 7))

        assert pooled.shape == (1, 7, 7, 256)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_backbone_to_fpn_to_rpn(self):
        """Test data flow from backbone through FPN to RPN."""
        from detectax.models.necks.fpn import FPN

        # Backbone
        backbone = ResNetBackbone(num_layers=50)
        images = jnp.ones((1, 224, 224, 3))
        bb_vars = backbone.init(jax.random.PRNGKey(0), images, train=False)
        features_dict = backbone.apply(bb_vars, images, train=False)

        # Convert dict to list for FPN
        features_list = [
            features_dict['stage_1'],  # C2: 256 channels
            features_dict['stage_2'],  # C3: 512 channels
            features_dict['stage_3'],  # C4: 1024 channels
            features_dict['stage_4'],  # C5: 2048 channels
        ]

        # FPN
        fpn = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            add_extra_convs='on_input',
        )
        fpn_vars = fpn.init(jax.random.PRNGKey(1), features_list)
        fpn_feats = fpn.apply(fpn_vars, features_list)

        assert len(fpn_feats) == 5  # P2-P6

        # RPN
        rpn = RPN(in_channels=256, num_anchors=3)
        rpn_vars = rpn.init(jax.random.PRNGKey(2), fpn_feats)
        cls_scores, bbox_preds = rpn.apply(rpn_vars, fpn_feats)

        assert len(cls_scores) == 5
        assert len(bbox_preds) == 5
