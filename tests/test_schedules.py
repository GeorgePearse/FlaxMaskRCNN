"""Tests for Detectax learning rate schedules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from detectax.training import (
    CosineDecayConfig,
    LearningRateScheduleConfig,
    StepDecayConfig,
    WarmupConfig,
    create_learning_rate_schedule,
)

jax = pytest.importorskip("jax")
pytest.importorskip("optax")


def test_linear_warmup_progression() -> None:
    """Warmup should ramp linearly to the base learning rate."""
    config = LearningRateScheduleConfig(
        schedule="step_decay",
        base_learning_rate=0.02,
        steps_per_epoch=1000,
        total_epochs=1,
        warmup=WarmupConfig(steps=500),
        step_decay=StepDecayConfig(milestones=(8, 11), decay_factor=0.1),
    )
    schedule = create_learning_rate_schedule(config)

    np.testing.assert_allclose(float(schedule(0)), 0.0, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(250)), 0.02 * 0.5, rtol=1e-6)
    expected_last_warmup = 0.02 * (499 / 500)
    np.testing.assert_allclose(float(schedule(499)), expected_last_warmup, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(float(schedule(500)), 0.02, rtol=1e-6)


def test_step_decay_milestones_apply_multiplicatively() -> None:
    """Step decay should multiply the base rate at milestone boundaries."""
    config = LearningRateScheduleConfig(
        schedule="step_decay",
        base_learning_rate=0.02,
        steps_per_epoch=1000,
        total_epochs=12,
        warmup=WarmupConfig(steps=500),
        step_decay=StepDecayConfig(milestones=(8, 11), decay_factor=0.1),
    )
    schedule = create_learning_rate_schedule(config)

    before_first_drop = config.steps_per_epoch * 8 - 1
    first_drop_step = config.steps_per_epoch * 8
    second_drop_step = config.steps_per_epoch * 11

    np.testing.assert_allclose(float(schedule(before_first_drop)), 0.02, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(first_drop_step)), 0.002, rtol=1e-6)
    np.testing.assert_allclose(float(schedule(second_drop_step)), 0.0002, rtol=1e-6)


def test_cosine_schedule_reaches_alpha_at_final_step() -> None:
    """Cosine decay should converge to the configured alpha multiplier."""
    config = LearningRateScheduleConfig(
        schedule="cosine",
        base_learning_rate=0.01,
        steps_per_epoch=10,
        total_epochs=5,
        warmup=WarmupConfig(steps=5),
        cosine=CosineDecayConfig(alpha=0.2),
    )
    schedule = create_learning_rate_schedule(config)

    assert float(schedule(0)) == pytest.approx(0.0, rel=1e-6)
    assert float(schedule(config.warmup.steps)) == pytest.approx(0.01, rel=1e-6)

    decay_steps = config.total_steps() - config.warmup.steps - 1
    mid_step = config.warmup.steps + decay_steps // 2
    progress = (mid_step - config.warmup.steps) / decay_steps
    cosine_component = 0.5 * (1.0 + math.cos(math.pi * progress))
    expected_mid = 0.01 * ((1 - config.cosine.alpha) * cosine_component + config.cosine.alpha)

    assert float(schedule(mid_step)) == pytest.approx(expected_mid, rel=1e-6)
    final_step = config.total_steps() - 1
    assert float(schedule(final_step)) == pytest.approx(0.01 * config.cosine.alpha, rel=1e-6)
