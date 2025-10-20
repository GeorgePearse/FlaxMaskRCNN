"""Training loop utilities for Detectax Mask R-CNN models.

This module provides a configurable training loop built on top of JAX/Flax.
It supports single-host multi-device (pmap) execution, gradient clipping with
the AdamW optimizer, Orbax checkpointing, and tqdm-powered logging. The entry
point is :func:`train`, which expects an :class:`ml_collections.ConfigDict`
describing the experiment setup (model construction, data pipeline, optimizer
configuration, and training hyperparameters).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils, struct
from flax.core import FrozenDict, freeze
from jax import tree_util
from orbax import checkpoint as ocp
from tqdm.auto import tqdm

from ml_collections import ConfigDict

Array = jax.Array
PyTree = Any
Batch = Mapping[str, PyTree]
LossFn = Callable[
    [FrozenDict[str, Any], Batch, FrozenDict[str, Any] | None, Array, Callable[..., Any]],
    tuple[Array, Mapping[str, Array], FrozenDict[str, Any] | None],
]

__all__ = ["TrainState", "TrainingResult", "train"]


@struct.dataclass
class TrainState:
    """Container for training state replicated across devices."""

    step: Array
    params: FrozenDict[str, Any]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    apply_fn: Callable[..., Any] = struct.field(pytree_node=False)
    model_state: FrozenDict[str, Any] | None = None
    rng: Array | None = None

    def apply_gradients(self, *, grads: FrozenDict[str, Any], model_state: FrozenDict[str, Any] | None) -> TrainState:
        """Applies gradients and updates optional mutable model state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + jnp.array(1, dtype=jnp.int32),
            params=new_params,
            opt_state=new_opt_state,
            model_state=model_state if model_state is not None else self.model_state,
        )

    def to_state_dict(self) -> dict[str, PyTree]:
        """Serialises the minimal training state required for checkpointing."""
        return {
            "step": self.step,
            "params": self.params,
            "opt_state": self.opt_state,
            "model_state": self.model_state,
            "rng": self.rng,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, PyTree], *, tx: optax.GradientTransformation, apply_fn: Callable[..., Any]) -> TrainState:
        """Restores a :class:`TrainState` from a checkpoint payload."""
        model_state = state_dict.get("model_state")
        if model_state is not None and not isinstance(model_state, FrozenDict):
            model_state = freeze(model_state)
        return cls(
            step=state_dict["step"],
            params=freeze(state_dict["params"]),
            opt_state=state_dict["opt_state"],
            tx=tx,
            apply_fn=apply_fn,
            model_state=model_state,
            rng=state_dict.get("rng"),
        )


@dataclass
class TrainingResult:
    """Summarises the outcome of a training run."""

    state: TrainState
    history: list[dict[str, float]]


def _as_jnp(tree: PyTree) -> PyTree:
    """Converts a pytree of array-likes to JAX arrays."""
    return tree_util.tree_map(
        lambda x: jnp.asarray(x) if np.isscalar(x) or isinstance(x, np.ndarray | jnp.ndarray | Array) else x,
        tree,
        is_leaf=lambda x: isinstance(x, np.ndarray | jnp.ndarray | Array) or np.isscalar(x),
    )


def _shard_batch(batch: Batch, num_devices: int) -> Batch:
    """Reshapes a global batch for pmap execution."""

    def _reshape(x: PyTree) -> PyTree:
        arr = jnp.asarray(x)
        if arr.shape[0] % num_devices != 0:
            msg = f"Global batch dimension {arr.shape[0]} is not divisible by num_devices={num_devices}."
            raise ValueError(msg)
        new_shape = (num_devices, arr.shape[0] // num_devices, *arr.shape[1:])
        return arr.reshape(new_shape)

    return jax.tree_map(_reshape, batch)


def _maybe_get_schedule(learning_rate: float | int | Callable[[int], float]) -> Callable[[int], float]:
    """Normalises a learning-rate specification to an Optax schedule."""
    if isinstance(learning_rate, float | int):
        return optax.constant_schedule(float(learning_rate))
    if callable(learning_rate):
        return learning_rate
    msg = "optimizer.learning_rate must be a float or callable schedule."
    raise TypeError(msg)


def _create_optimizer(cfg: ConfigDict, params: FrozenDict[str, Any]) -> optax.GradientTransformation:
    """Builds the Optax optimizer pipeline (AdamW with optional clipping)."""
    if "learning_rate" not in cfg:
        raise ValueError("optimizer.learning_rate is required.")
    schedule = _maybe_get_schedule(cfg.learning_rate)
    adamw = optax.adamw(
        learning_rate=schedule,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    transforms: list[optax.GradientTransformation] = []
    clip_norm = cfg.get("clip_norm")
    if clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(float(clip_norm)))
    transforms.append(adamw)
    return optax.chain(*transforms)


def _default_loss_fn(
    params: FrozenDict[str, Any],
    batch: Batch,
    model_state: FrozenDict[str, Any] | None,
    rng: Array,
    apply_fn: Callable[..., Any],
    *,
    train_args: Sequence[Any],
    train_kwargs: Mapping[str, Any],
) -> tuple[Array, Mapping[str, Array], FrozenDict[str, Any] | None]:
    """Fallback loss that expects the model output to contain a ``loss`` scalar."""
    variables: dict[str, Any] = {"params": params}
    mutable = False
    if model_state:
        variables.update(model_state)
        mutable = list(model_state.keys())

    rngs = {"dropout": rng} if rng is not None else None
    if mutable:
        outputs, new_model_state = apply_fn(
            variables,
            *train_args,
            rngs=rngs,
            mutable=mutable,
            **batch,
            **train_kwargs,
        )
    else:
        outputs = apply_fn(
            variables,
            *train_args,
            rngs=rngs,
            **batch,
            **train_kwargs,
        )
        new_model_state = model_state

    if isinstance(outputs, Mapping) and "loss" in outputs:
        loss = outputs["loss"]
        metrics = {k: v for k, v in outputs.items() if k != "loss"}
    else:
        raise ValueError(
            "The default loss_fn expects the model to return a mapping containing a 'loss' entry. "
            "Provide a custom config.loss_fn for alternative behaviours."
        )

    if new_model_state is not None and not isinstance(new_model_state, FrozenDict):
        new_model_state = freeze(new_model_state)
    return loss, metrics, new_model_state


def _build_train_step(loss_fn: LossFn, axis_name: str | None) -> Callable[[TrainState, Batch], tuple[TrainState, Mapping[str, Array]]]:
    """Creates the per-step update function, optionally averaged across devices."""

    def train_step(state: TrainState, batch: Batch) -> tuple[TrainState, Mapping[str, Array]]:
        rng, step_rng = jax.random.split(state.rng) if state.rng is not None else (None, None)

        def loss_with_aux(params: FrozenDict[str, Any]) -> tuple[Array, tuple[Mapping[str, Array], FrozenDict[str, Any] | None]]:
            loss, metrics, new_model_state = loss_fn(params, batch, state.model_state, step_rng, state.apply_fn)
            return loss, (metrics, new_model_state)

        (loss, (metrics, new_model_state)), grads = jax.value_and_grad(loss_with_aux, has_aux=True)(state.params)
        if axis_name is not None:
            grads = jax.lax.pmean(grads, axis_name=axis_name)

        new_state = state.apply_gradients(grads=grads, model_state=new_model_state)
        new_state = new_state.replace(rng=rng)

        if axis_name is not None:
            loss = jax.lax.pmean(loss, axis_name=axis_name)
            metrics = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name=axis_name), metrics)

        updated_metrics = dict(metrics)
        updated_metrics["loss"] = loss
        return new_state, updated_metrics

    return train_step


def _prepare_loss_fn(config: ConfigDict, train_kwargs: Mapping[str, Any]) -> LossFn:
    """Returns the configured loss function, falling back to the default."""
    maybe_loss_fn = config.get("loss_fn")
    if maybe_loss_fn is not None:
        return maybe_loss_fn
    train_args = tuple(config.get("model", {}).get("train_args", ()))
    return partial(_default_loss_fn, train_args=train_args, train_kwargs=train_kwargs)


def _initialise_state(config: ConfigDict, rng: Array) -> TrainState:
    """Constructs the initial training state from the configuration."""
    if "model" not in config or "module" not in config.model:
        raise ValueError("config.model.module must be provided.")
    model = config.model.module

    init_args = tuple(config.model.get("init_args", ()))
    init_kwargs = dict(config.model.get("init_kwargs", {}))
    init_args = tuple(_as_jnp(arg) for arg in init_args)
    init_kwargs = {k: _as_jnp(v) for k, v in init_kwargs.items()}

    init_rng, state_rng = jax.random.split(rng)
    variables = model.init(init_rng, *init_args, **init_kwargs)
    params = freeze(variables["params"])

    non_param_state = {k: v for k, v in variables.items() if k != "params"}
    model_state = freeze(non_param_state) if non_param_state else None

    optimizer = _create_optimizer(config.optimizer, params)
    opt_state = optimizer.init(params)
    return TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=opt_state,
        tx=optimizer,
        apply_fn=model.apply,
        model_state=model_state,
        rng=state_rng,
    )


def _setup_checkpointing(checkpoint_cfg: ConfigDict | None) -> tuple[ocp.PyTreeCheckpointer | None, Path | None, int]:
    """Creates checkpointing utilities if enabled."""
    if not checkpoint_cfg or "dir" not in checkpoint_cfg:
        return None, None, 0
    ckpt_dir = Path(checkpoint_cfg.dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    keep = int(checkpoint_cfg.get("max_to_keep", 1))
    return ocp.PyTreeCheckpointer(), ckpt_dir, keep


def _latest_checkpoint_path(ckpt_dir: Path) -> Path | None:
    checkpoints = sorted((p for p in ckpt_dir.iterdir() if p.is_dir()), key=lambda path: path.name)
    return checkpoints[-1] if checkpoints else None


def _restore_state(
    state: TrainState, checkpointer: ocp.PyTreeCheckpointer | None, ckpt_dir: Path | None, checkpoint_cfg: ConfigDict | None
) -> TrainState:
    """Restores a checkpoint if available and requested."""
    if checkpointer is None or ckpt_dir is None or not checkpoint_cfg or not checkpoint_cfg.get("restore", False):
        return state
    latest = _latest_checkpoint_path(ckpt_dir)
    if latest is None:
        return state
    payload = checkpointer.restore(str(latest))
    return TrainState.from_state_dict(payload, tx=state.tx, apply_fn=state.apply_fn)


def _save_checkpoint(
    checkpointer: ocp.PyTreeCheckpointer | None,
    ckpt_dir: Path | None,
    max_to_keep: int,
    step: int,
    state: TrainState,
) -> None:
    if checkpointer is None or ckpt_dir is None:
        return
    ckpt_path = ckpt_dir / f"step_{step:08d}"
    checkpointer.save(str(ckpt_path), state.to_state_dict())
    if max_to_keep > 0:
        checkpoints = sorted((p for p in ckpt_dir.iterdir() if p.is_dir()), key=lambda path: path.name)
        excess = len(checkpoints) - max_to_keep
        for old in checkpoints[: max(0, excess)]:
            shutil.rmtree(old, ignore_errors=True)


def train(config: ConfigDict) -> TrainingResult:
    """Runs the training loop as described by ``config``.

    The configuration dictionary should contain the following sections:

    - ``seed`` (int, optional): RNG seed, defaults to 0.
    - ``model``: fields ``module`` (Flax module instance), ``init_args`` /
      ``init_kwargs`` (inputs for ``Module.init``), optional ``train_args`` /
      ``train_kwargs`` passed to ``Module.apply`` during training.
    - ``optimizer``: at minimum ``learning_rate`` (float or schedule). Optional
      ``beta1``, ``beta2``, ``eps``, ``weight_decay``, ``clip_norm``.
    - ``data``: provides ``train_iter_fn`` returning an iterator over batches.
    - ``training``: fields ``num_steps`` (int), optional ``log_every_steps``,
      ``checkpoint_every_steps``, ``use_tqdm``.
    - ``loss_fn`` (optional): overrides the default loss computation.
    - ``checkpoint`` (optional): ``dir`` for saving checkpoints alongside
      ``max_to_keep`` and ``restore`` flags.

    Returns:
        :class:`TrainingResult` containing the final (unreplicated) train state
        and a list of metric dictionaries logged at every step.
    """
    if "data" not in config or "train_iter_fn" not in config.data:
        raise ValueError("config.data.train_iter_fn must be provided.")
    if "optimizer" not in config:
        raise ValueError("config.optimizer must be provided.")
    if "training" not in config or "num_steps" not in config.training:
        raise ValueError("config.training.num_steps must be specified.")

    rng = jax.random.PRNGKey(config.get("seed", 0))
    state = _initialise_state(config, rng)

    checkpoint_cfg = config.get("checkpoint")
    checkpointer, ckpt_dir, max_to_keep = _setup_checkpointing(checkpoint_cfg)
    state = _restore_state(state, checkpointer, ckpt_dir, checkpoint_cfg)

    num_devices = jax.local_device_count()
    axis_name = "devices" if num_devices > 1 else None

    train_kwargs = dict(config.get("model", {}).get("train_kwargs", {}))
    loss_fn = _prepare_loss_fn(config, train_kwargs)
    train_step_fn = _build_train_step(loss_fn, axis_name)
    compiled_train_step: Callable[[TrainState, Batch], tuple[TrainState, Mapping[str, Array]]]

    if num_devices > 1:
        compiled_train_step = jax.pmap(train_step_fn, axis_name=axis_name)
        state = jax_utils.replicate(state)

        def prepare_batch(batch):
            return _shard_batch(_as_jnp(batch), num_devices)

        def unreplicate(value):
            return jax_utils.unreplicate(value)
    else:
        compiled_train_step = jax.jit(train_step_fn)

        def prepare_batch(batch):
            return _as_jnp(batch)

        def unreplicate(value):
            return value

    train_iterator: Iterator[Batch] = config.data.train_iter_fn()
    num_steps = int(config.training.num_steps)
    log_every = int(config.training.get("log_every_steps", 1))
    checkpoint_every = int(config.training.get("checkpoint_every_steps", num_steps))
    use_tqdm = bool(config.training.get("use_tqdm", True))

    start_step = int(jax.device_get(unreplicate(state).step))
    history: list[dict[str, float]] = []

    progress = tqdm(
        range(start_step, num_steps),
        initial=start_step,
        total=num_steps,
        disable=not use_tqdm,
        desc="training",
        leave=False,
    )

    for step in progress:
        raw_batch = next(train_iterator)
        batch = prepare_batch(raw_batch)
        state, metrics = compiled_train_step(state, batch)

        host_metrics = jax.device_get(metrics)
        host_metrics = unreplicate(host_metrics)
        host_metrics = {k: float(np.array(v)) for k, v in host_metrics.items()}
        history.append(host_metrics)

        step_index = step + 1
        if log_every and step_index % log_every == 0:
            progress.set_postfix({k: f"{v:.4f}" for k, v in host_metrics.items()})

        if checkpoint_every and (step_index % checkpoint_every == 0 or step_index == num_steps):
            state_to_save = unreplicate(state)
            _save_checkpoint(checkpointer, ckpt_dir, max_to_keep, step_index, state_to_save)

    final_state = unreplicate(state)
    return TrainingResult(state=final_state, history=history)
