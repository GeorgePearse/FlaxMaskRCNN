"""Tests for the Detectax training loop."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from detectrax.training import train as training_lib
from ml_collections import ConfigDict


class _DummyModel(nn.Module):
    """A tiny linear model for smoke-testing the training loop."""

    @nn.compact
    def __call__(self, inputs: jax.Array, train: bool = False) -> jax.Array:  # noqa: ARG002 (train flag for API parity)
        dense = nn.Dense(features=1, use_bias=False)
        return dense(inputs)


def _make_iterator(batch_size: int = 4) -> Iterator[dict[str, jax.Array]]:
    """Creates a simple iterator yielding constant batches."""

    inputs = jnp.ones((batch_size, 4), dtype=jnp.float32)
    targets = jnp.zeros((batch_size, 1), dtype=jnp.float32)

    def _generator() -> Iterator[dict[str, jax.Array]]:
        while True:
            yield {"inputs": inputs, "targets": targets}

    return _generator()


def _loss_fn(
    params: Any,
    batch: dict[str, jax.Array],
    model_state: Any,
    rng: jax.Array | None,
    apply_fn: Any,
):
    """Mean-squared error loss used in the smoke test."""
    del model_state, rng  # Unused in this simple loss.
    variables = {"params": params}
    preds = apply_fn(variables, batch["inputs"], train=True)
    loss = jnp.mean((preds - batch["targets"]) ** 2)
    metrics = {"mse": loss}
    return loss, metrics, None


def test_training_smoke(tmp_path: Path) -> None:
    """Runs a single training step to ensure the loop is wired correctly."""
    model = _DummyModel()

    config = ConfigDict()
    config.seed = 0

    config.model = ConfigDict()
    config.model.module = model
    config.model.init_args = [jnp.ones((4, 4), dtype=jnp.float32)]
    config.model.init_kwargs = {"train": False}

    config.optimizer = ConfigDict()
    config.optimizer.learning_rate = 1e-3
    config.optimizer.clip_norm = 1.0

    config.data = ConfigDict()
    config.data.train_iter_fn = lambda: _make_iterator(batch_size=4)

    config.training = ConfigDict()
    config.training.num_steps = 1
    config.training.log_every_steps = 1
    config.training.use_tqdm = False
    config.training.checkpoint_every_steps = 1

    config.checkpoint = ConfigDict()
    config.checkpoint.dir = str(tmp_path / "checkpoints")
    config.checkpoint.max_to_keep = 1
    config.checkpoint.restore = False

    config.loss_fn = _loss_fn

    result = training_lib.train(config)

    assert result.state.step.item() == 1
    assert result.history, "Expected non-empty training history."
    final_loss = result.history[-1]["loss"]
    assert bool(jnp.isfinite(final_loss))

    checkpoint_dir = Path(config.checkpoint.dir)
    saved_steps = sorted(p.name for p in checkpoint_dir.iterdir() if p.is_dir())
    assert saved_steps, "Expected a checkpoint directory to be created."
