# Why JAX for Detectax?

Detectax is built on the JAX ecosystem, which provides powerful tools for high-performance machine learning research and production.

## JAX Ecosystem

Detectax leverages these amazing JAX libraries:

### Deep Learning
- **[Flax](https://github.com/google/flax)** - Neural network library ✅ *Core framework*
- **[Equinox](https://github.com/patrick-kidger/equinox)** - Neural networks and PyTree utilities
- **[Optax](https://github.com/deepmind/optax)** - Gradient optimizers (SGD, Adam, ...) ✅ *Already using*
- **[Orbax](https://github.com/google/orbax)** - Checkpointing (async/multi-host) ✅ *Already using*
- **[Scenic](https://github.com/google-research/scenic)** - Training infrastructure ✅ *Core infrastructure*
- **[Levanter](https://github.com/stanford-crfm/levanter)** - Scalable training of foundation models
- **[paramax](https://github.com/patrick-kidger/paramax)** - Parameterizations and constraints for PyTrees

### Scientific Computing
- **[Diffrax](https://github.com/patrick-kidger/diffrax)** - Numerical differential equation solvers
- **[Optimistix](https://github.com/patrick-kidger/optimistix)** - Root finding, minimization, least squares
- **[Lineax](https://github.com/patrick-kidger/lineax)** - Linear solvers
- **[BlackJAX](https://github.com/blackjax-devs/blackjax)** - Probabilistic & Bayesian sampling
- **[sympy2jax](https://github.com/google/sympy2jax)** - SymPy ↔ JAX conversion
- **[PySR](https://github.com/MilesCranmer/PySR)** - Symbolic regression (non-JAX but awesome!)

## Why JAX?

1. **Functional Programming**: Clean, composable code with no hidden state
2. **Performance**: JIT compilation for near-C++ speeds in Python
3. **Automatic Differentiation**: Powerful `grad`, `jacobian`, `hessian` transforms
4. **Parallelization**: Easy multi-GPU/TPU with `pmap`, `pjit`
5. **Reproducibility**: Explicit random key management ensures reproducible experiments
6. **Composability**: `jax.vmap`, `jax.jit`, `jax.grad` compose seamlessly

## More JAX Projects

See **[Awesome JAX](https://github.com/n2cholas/awesome-jax)** for a comprehensive list of JAX projects and resources!

## Comparison with PyTorch

| Feature | JAX/Flax | PyTorch |
|---------|----------|---------|
| Programming Model | Functional | Object-oriented |
| JIT Compilation | `jax.jit` (XLA) | `torch.jit` (TorchScript) |
| Auto-diff | Functional transforms | Autograd with tape |
| Multi-device | `pmap`, `pjit` | DistributedDataParallel |
| Random Numbers | Explicit PRNG keys | Global random state |
| Ecosystem | Growing rapidly | Mature and extensive |

JAX's functional approach and XLA compilation make it ideal for research that requires flexibility and performance at scale.
