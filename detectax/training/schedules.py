"""Learning rate schedule utilities for Detectax training loops.

This module provides composable helpers to create Optax-compatible learning
rate schedules driven by configuration objects. The default configuration
implements the canonical Mask R-CNN training recipe with:

* Linear warmup for the first 500 optimization steps.
* Step decay that multiplies the base rate by ``0.1`` at the start of epochs 8
  and 11 for a 12-epoch schedule.
* Optional cosine annealing after warmup.

All schedules return scalar ``float32`` values and are safe to jit or pmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

ScheduleType = Literal["step_decay", "cosine"]


class ScheduleFn(Protocol):
    """Callable signature for Optax schedules."""

    def __call__(self, step: int | jnp.ndarray) -> Float[Array, ""]:
        raise NotImplementedError


@dataclass(frozen=True)
class WarmupConfig:
    """Configuration for the warmup phase."""

    steps: int = 500
    strategy: Literal["linear"] = "linear"


@dataclass(frozen=True)
class StepDecayConfig:
    """Configuration for piecewise-constant (step) decay."""

    milestones: tuple[int, ...] = (8, 11)
    decay_factor: float = 0.1


@dataclass(frozen=True)
class CosineDecayConfig:
    """Configuration for cosine annealing."""

    alpha: float = 0.0  # final learning rate multiplier


@dataclass(frozen=True)
class LearningRateScheduleConfig:
    """Complete configuration for Detectax learning rate schedules."""

    schedule: ScheduleType = "step_decay"
    base_learning_rate: float = 0.02
    steps_per_epoch: int = 1000
    total_epochs: int = 12
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    step_decay: StepDecayConfig = field(default_factory=StepDecayConfig)
    cosine: CosineDecayConfig = field(default_factory=CosineDecayConfig)

    def total_steps(self) -> int:
        """Total number of optimization steps for the configured run."""
        return self.steps_per_epoch * self.total_epochs


def create_learning_rate_schedule(config: LearningRateScheduleConfig) -> ScheduleFn:
    """Instantiate a learning rate schedule based on the provided configuration.

    Args:
        config: Learning rate configuration describing warmup and decay.

    Returns:
        An Optax-compatible schedule callable that maps a step index to a scalar
        learning rate (float32).
    """
    _validate_config(config)

    if config.schedule == "step_decay":
        base_schedule = _create_step_decay_schedule(config)
    elif config.schedule == "cosine":
        base_schedule = _create_cosine_schedule(config)
    else:  # pragma: no cover - guarded by _validate_config
        raise ValueError(f"Unsupported schedule type: {config.schedule}")

    if config.warmup.steps <= 0:
        return base_schedule

    return _apply_linear_warmup(base_schedule, warmup_steps=config.warmup.steps)


def _validate_config(config: LearningRateScheduleConfig) -> None:
    if config.base_learning_rate <= 0.0:
        raise ValueError("base_learning_rate must be positive.")
    if config.steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive.")
    if config.total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if config.warmup.steps < 0:
        raise ValueError("warmup.steps must be non-negative.")
    if config.warmup.strategy != "linear":
        raise ValueError(f"Unsupported warmup strategy: {config.warmup.strategy}")
    if config.schedule not in ("step_decay", "cosine"):
        raise ValueError(f"Unsupported schedule type: {config.schedule}")
    if config.step_decay.decay_factor <= 0.0 or config.step_decay.decay_factor >= 1.0:
        raise ValueError("step_decay.decay_factor must be in (0, 1).")
    if any(epoch <= 0 for epoch in config.step_decay.milestones):
        raise ValueError("step_decay.milestones must contain positive epoch indices.")
    if config.cosine.alpha < 0.0 or config.cosine.alpha > 1.0:
        raise ValueError("cosine.alpha must be within [0, 1].")


def _create_step_decay_schedule(config: LearningRateScheduleConfig) -> ScheduleFn:
    boundaries: dict[int, float] = {}
    total_steps = config.total_steps()

    for epoch in sorted(config.step_decay.milestones):
        boundary_step = epoch * config.steps_per_epoch
        if boundary_step >= total_steps:
            # Ignore milestones that occur at or after training ends.
            continue
        boundaries[boundary_step] = config.step_decay.decay_factor

    piecewise = optax.piecewise_constant_schedule(
        init_value=config.base_learning_rate,
        boundaries_and_scales=boundaries,
    )

    def schedule(step: int | jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(piecewise(step), dtype=jnp.float32)

    return schedule


def _create_cosine_schedule(config: LearningRateScheduleConfig) -> ScheduleFn:
    warmup_steps = config.warmup.steps
    total_steps = config.total_steps()
    decay_steps = max(total_steps - warmup_steps - 1, 1)

    cosine = optax.cosine_decay_schedule(
        init_value=config.base_learning_rate,
        decay_steps=decay_steps,
        alpha=config.cosine.alpha,
    )

    def schedule(step: int | jnp.ndarray) -> jnp.ndarray:
        step_arr = jnp.asarray(step, dtype=jnp.int32)
        post_warmup = jnp.maximum(step_arr - warmup_steps, 0)
        return jnp.asarray(cosine(post_warmup), dtype=jnp.float32)

    return schedule


def _apply_linear_warmup(schedule: ScheduleFn, *, warmup_steps: int) -> ScheduleFn:
    warmup_steps = int(warmup_steps)
    warmup_denominator = jnp.asarray(max(warmup_steps, 1), dtype=jnp.float32)

    def wrapped(step: int | jnp.ndarray) -> jnp.ndarray:
        step_arr = jnp.asarray(step)
        base_lr = schedule(step_arr)
        step_float = step_arr.astype(jnp.float32)
        warmup_progress = jnp.minimum(step_float / warmup_denominator, 1.0)
        warmed = base_lr * warmup_progress
        return jnp.where(step_arr < warmup_steps, warmed, base_lr)

    return wrapped


__all__ = [
    "LearningRateScheduleConfig",
    "ScheduleFn",
    "WarmupConfig",
    "StepDecayConfig",
    "CosineDecayConfig",
    "create_learning_rate_schedule",
]
