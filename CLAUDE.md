# FlaxMaskRCNN Project Guidelines

## Purpose

Production-ready Mask R-CNN implementation in JAX/Flax using Google Scenic's training infrastructure.

## Autonomous Implementation Mode

**CRITICAL: Claude Code operates in autonomous completion mode for this project.**

- **Continue until exhaustion**: Keep implementing architecture tasks from ARCHITECTURE_TASKS.md until ALL possible components are complete
- **Do NOT stop to ask for next steps**: Autonomously proceed through all sprints (1-7) without user prompts
- **Verification gate**: Only stop work after consulting Gemini 2.5 Pro via clink: "Have I implemented absolutely every part of the Mask R-CNN JAX model that I possibly could?"
- **If answer is NO**: Continue implementing remaining components
- **If answer is YES**: Only then report completion to user
- **Maximum parallelism**: Launch as many independent Codex tasks as possible simultaneously
- **Continuous monitoring**: Review implementations as they complete, fix issues, validate tests, commit progress

## Technology Stack

- **Language**: Python 3.12
- **ML Framework**: JAX 0.4.35, Flax 0.10.2
- **Training Infrastructure**: Google Scenic (to be integrated from https://github.com/google-research/scenic)
- **Optimization**: Optax 0.2.4
- **Data Loading**: TensorFlow 2.18.0, tensorflow-datasets 4.9.7
- **Evaluation**: pycocotools 2.0.8
- **Type Checking**: Pyright, mypy, jaxtyping
- **Testing**: Pytest
- **Dependency Management**: uv

## Project Structure

```
flax_mask_rcnn/
├── models/
│   ├── backbones/       # Feature extractors (ResNet, ViT from Scenic)
│   ├── necks/           # FPN implementation
│   ├── heads/           # RPN, detection, mask heads
│   └── detectors/       # Complete Mask R-CNN model
├── data/                # Data loading and augmentation
├── configs/             # YAML configuration files
├── evaluation/          # COCO evaluation metrics
├── training/            # Training loops, optimizers, schedules
└── utils/               # Utilities (checkpointing, logging, etc.)
tests/                   # Unit and integration tests
reference/               # PyTorch visdet reference code (for architecture understanding)
```

## Key Principles

### 1. JAX/Flax Best Practices

- **Functional Programming**: All model code must be functional and pure
- **Explicit State**: Use Flax's Module system, no hidden mutable state
- **Pytrees**: Understand and use JAX pytrees for parameters and state
- **Transformations**: Use jax.jit, jax.vmap, jax.pmap appropriately
- **Random Keys**: Always pass and split PRNGKeys explicitly
- **No Side Effects**: Avoid print statements inside jitted functions

### 2. Scenic Integration

- **Follow Scenic Patterns**: Study and follow Scenic's architectural patterns
- **Use Scenic Base Classes**: Extend Scenic's base model and trainer classes
- **Reuse Components**: Leverage Scenic's existing backbones, FPN, and utilities
- **Training Infrastructure**: Use Scenic's distributed training, checkpointing, and metrics
- **Configuration**: Use ml_collections.ConfigDict for all configs

### 3. Type Safety

- **Complete Type Annotations**: All functions must have full type hints
- **jaxtyping**: Use jaxtyping for array shape annotations (Float[Array, "batch height width channels"])
- **No `Any` Types**: Avoid Any unless absolutely necessary and documented
- **Strict Mode**: Code must pass pyright in strict mode
- **Runtime Checks**: Use chex for runtime shape/type assertions during development

### 4. Code Quality

- **Never Simulate**: Never create fake losses or simulated training metrics
- **Full Implementation**: Always implement complete functionality, no simplified versions
- **Delete and Restart**: If code doesn't work, delete it and start fresh instead of patching
- **No System Path Hacks**: Fix underlying import issues instead of appending to sys.path
- **Line Length**: 150 characters maximum
- **Imports**: Use absolute imports within package

### 5. Testing

- **Test Data**: Use CMR COCO dataset for testing
  - Train: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
  - Val: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`
  - Images: `/home/georgepearse/data/images`
- **Unit Tests**: Test individual components (FPN, RPN, heads) in isolation
- **Integration Tests**: Test full model forward pass on small batches
- **Shape Tests**: Verify all tensor shapes throughout the model
- **Class Count**: Always verify annotation class counts match model architecture

### 6. Training

- **Progress Bars**: Always use tqdm for long-running operations
- **Structured Logging**: Use Python's logging module with clear format
- **Checkpointing**: Save checkpoints regularly using orbax-checkpoint
- **Reproducibility**: Set random seeds for reproducibility
- **Configuration**: All hyperparameters must be in YAML configs validated with ml_collections

### 7. Data Pipeline

- **COCO Format**: Follow COCO JSON annotation format
- **Augmentation**: Implement augmentations compatible with JAX (or use TF ops)
- **Validation**: Validate data formats and annotations early in the pipeline
- **Efficiency**: Use tf.data or grain for efficient data loading with prefetching
- **Mask Formats**: Ensure mask formats are consistent throughout (binary masks as arrays)

### 8. Reference Code Usage

- **Keep Reference Visible**: PyTorch visdet code in `reference/` directory provides architecture details
- **Port, Don't Copy**: Understand PyTorch code, then implement in JAX/Flax idiomatically
- **Document Differences**: Note where JAX implementation differs from PyTorch reference
- **Architecture Fidelity**: Match the mathematical operations, not the imperative style

## Common Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest -v

# Run type checking
uv run pyright

# Run linting
uv run ruff check .

# Format code
uv run black .

# Run training (to be implemented)
uv run python -m flax_mask_rcnn.training.train --config configs/mask_rcnn_r50_fpn.yaml
```

## Implementation Phases

1. **Phase 1**: Research Scenic, copy reference code from visdet
2. **Phase 2**: Implement core model (backbone, FPN, RPN, heads)
3. **Phase 3**: Port data pipeline from visdet
4. **Phase 4**: Implement training infrastructure with Scenic
5. **Phase 5**: Add evaluation and testing
6. **Phase 6**: Documentation and finalization

## Important Notes

- **Scenic Repository**: Clone https://github.com/google-research/scenic into the project or reference externally
- **GPU Support**: JAX with CUDA 12 support for GPU training
- **Distributed Training**: Use JAX's pmap/pjit for multi-GPU training
- **No PyTorch**: This is a pure JAX/Flax implementation, PyTorch visdet code is reference only

## Multi-Model Collaboration with Clink

This project uses the zen-mcp-server's `clink` tool to enable collaboration between different AI CLI agents (Claude Code, Codex CLI, Gemini CLI). This allows you to leverage different models' strengths while maintaining context.

### Why Use Clink with Codex?

**Codex is optimized for code generation and implementation tasks.** When you need to:
- Implement complex code structures
- Port algorithms from reference implementations
- Generate boilerplate with high precision
- Handle intricate type systems

**Use Codex via clink instead of doing it yourself.** This provides:
- **Context Isolation**: Heavy implementation tasks run in fresh context, preserving your working context window
- **Specialized Expertise**: Codex's code generation strengths for implementation-heavy work
- **Seamless Handoff**: Full conversation context is preserved when delegating tasks

### How to Use Clink

**Basic Usage:**

```python
# Delegate code implementation to Codex
mcp__zen__clink(
    prompt="Implement the RoI Align layer in JAX/Flax with proper type annotations",
    cli_name="codex",
    files=["/home/georgepearse/FlaxMaskRCNN/detectrax/models/layers/roi_align.py"]
)
```

**With Roles:**

```python
# Use Codex for code review
mcp__zen__clink(
    prompt="Review this FPN implementation for correctness and JAX best practices",
    cli_name="codex",
    role="codereviewer",
    files=["/home/georgepearse/FlaxMaskRCNN/detectrax/models/necks/fpn.py"]
)
```

**Available Roles:**
- `default`: General purpose coding
- `planner`: Architecture and design planning
- `codereviewer`: Code review and quality checks

### When to Use Clink

**DO use clink for:**
- ✅ Implementing complex algorithms from reference code
- ✅ Generating large code structures (full modules, classes)
- ✅ Porting PyTorch reference code to JAX/Flax
- ✅ Code reviews of completed implementations
- ✅ Deep debugging sessions that need fresh context

**DON'T use clink for:**
- ❌ Simple questions or clarifications (use chat tool instead)
- ❌ Quick edits or small changes
- ❌ Exploratory discussions (use consensus tool instead)
- ❌ File searches or code navigation

### Security Note

Clink spawns autonomous CLI agents with relaxed permissions (`--dangerously-bypass-approvals-and-sandbox` for Codex). These agents can edit files and run commands. Only use clink when you want full autonomous implementation.

### Typical Workflow

1. **Plan with Claude Code** (you): Understand requirements, design architecture
2. **Implement with Codex** (clink): Heavy lifting of code generation
3. **Review with Claude Code** (you): Verify, test, iterate
4. **Consensus for decisions** (zen consensus tool): Debate architectural choices with multiple models

**Example:**

```python
# Step 1: You plan the approach
"Let's port the RPN head from reference/visdet_models to JAX/Flax"

# Step 2: Delegate implementation to Codex
mcp__zen__clink(
    prompt="""
    Port the RPN head from reference/visdet_models/dense_heads/anchor_head.py
    to detectrax/models/heads/rpn_head.py using JAX/Flax patterns.

    Requirements:
    - Use Flax nn.Module
    - Full type annotations with jaxtyping
    - Follow existing FPN implementation patterns
    - Include comprehensive docstrings
    """,
    cli_name="codex",
    files=[
        "reference/visdet_models/dense_heads/anchor_head.py",
        "detectrax/models/necks/fpn.py"  # for pattern reference
    ]
)

# Step 3: Review the implementation
"Let's review what Codex generated and run tests"
```

### Pro Tips

- **Always provide file paths**: Codex performs better with concrete file references
- **Be explicit about requirements**: Mention patterns to follow, style guidelines, type annotations
- **Reference existing code**: Point to similar implementations as examples
- **Use continuation_id**: Reuse continuation_id across clink calls to maintain context across multiple Codex sessions
- **Prefer Codex for porting**: When porting from PyTorch reference, Codex excels at maintaining semantic equivalence

## Background Task Orchestration Workflow

**CRITICAL: Claude Code's role in this project is TASK ORCHESTRATOR, not implementer.**

All architecture implementation tasks MUST be executed in background via Codex/clink. This maximizes throughput by running multiple tasks in parallel while Claude Code manages coordination, reviews implementations, and validates results.

### The Orchestration Model

**Claude Code Responsibilities:**
1. **Task Creation**: Break down architecture work into granular tasks (see ARCHITECTURE_TASKS.md)
2. **Parallel Assignment**: Launch as many independent tasks as possible simultaneously
3. **Progress Monitoring**: Track task completion via background bash processes
4. **Implementation Review**: Verify generated code against specs when tasks complete
5. **Quality Validation**: Run tests, check types, ensure correctness
6. **Integration**: Commit validated implementations and update progress tracking

**Codex Responsibilities (via clink):**
1. **Implementation**: Generate complete, production-ready code
2. **Testing**: Create comprehensive test suites
3. **Documentation**: Add docstrings and type annotations
4. **Autonomy**: Handle entire implementation lifecycle for assigned task

### Task Definition Format

All tasks are defined as JSON files in the `/tasks` directory (version controlled):

```json
{
  "prompt": "Implement Task X.Y: [Task Name] as described in ARCHITECTURE_TASKS.md.\n\nCreate [file path] for [purpose].\n\nRequirements:\n- [detailed requirements]\n- [technical specifications]\n- [edge cases]\n\nAlso create tests in [test file path]:\n- [test scenario 1]\n- [test scenario 2]\n\nUpdate [related files].\n\nReference: [reference implementations]",
  "cli_name": "codex",
  "files": [
    "/absolute/path/to/reference1.py",
    "/absolute/path/to/reference2.py"
  ]
}
```

**Key Principles:**
- **Absolute paths only**: All file references must be absolute
- **Complete specifications**: Include requirements, edge cases, test scenarios
- **Reference context**: Provide reference implementations for pattern matching
- **Self-contained**: Each task should be executable independently

### Parallel Execution Pattern

**Step 1: Create task definitions in /tasks directory**

```bash
# Example: Sprint 1 tasks
/tasks/sprint_1/task_1_2_box_coder.json
/tasks/sprint_1/task_1_3_nms.json
/tasks/sprint_1/task_1_4_iou.json
```

**Step 2: Launch tasks in parallel via background bash**

```bash
#!/bin/bash
cd /home/georgepearse/FlaxMaskRCNN

# Launch multiple Codex tasks in background
(cat /tasks/sprint_1/task_1_2_box_coder.json | codex exec --json --dangerously-bypass-approvals-and-sandbox > /tmp/task_1_2_output.json 2>&1) &
(cat /tasks/sprint_1/task_1_3_nms.json | codex exec --json --dangerously-bypass-approvals-and-sandbox > /tmp/task_1_3_output.json 2>&1) &
(cat /tasks/sprint_1/task_1_4_iou.json | codex exec --json --dangerously-bypass-approvals-and-sandbox > /tmp/task_1_4_output.json 2>&1) &

# PIDs captured for monitoring
echo "Tasks launched: Box Coder, NMS, IoU"
wait
echo "All tasks complete"
```

**Step 3: Review implementations as they complete**

```python
# Claude Code monitors task completion and reviews
# When task completes:
# 1. Read generated implementation files
# 2. Verify against task requirements
# 3. Run tests: uv run pytest tests/test_*.py -v
# 4. Check types: uv run pyright
# 5. If all pass: git add + commit
# 6. If failures: analyze, fix, or reassign to Codex
```

### Review Checklist

When tasks complete, Claude Code verifies:

✅ **Code Quality**
- [ ] All functions have complete type annotations (jaxtyping)
- [ ] Docstrings present and comprehensive
- [ ] Follows JAX/Flax functional patterns
- [ ] No mutable state or side effects

✅ **Correctness**
- [ ] Implementation matches ARCHITECTURE_TASKS.md specification
- [ ] All edge cases handled (zero-area boxes, empty inputs, etc.)
- [ ] Numerical stability considerations (epsilon for division, clipping)

✅ **Testing**
- [ ] All tests pass: `uv run pytest tests/test_*.py -v`
- [ ] Test coverage includes edge cases
- [ ] Tests verify shapes, gradients, and numerical correctness

✅ **Integration**
- [ ] Exports added to `__init__.py`
- [ ] No import errors or circular dependencies
- [ ] Type checking passes: `uv run pyright`

✅ **Documentation**
- [ ] ARCHITECTURE_TASKS.md progress updated
- [ ] Commit message references task number and spec

### Workflow Example: Sprint 1 (Tasks 1.2-1.4)

```python
# 1. Claude Code creates task definitions
# (Already done - see /tasks/sprint_1/*.json)

# 2. Launch parallel Codex tasks
Bash(command="bash /tasks/sprint_1/run_parallel.sh", run_in_background=True)

# 3. Monitor progress (Claude Code checks periodically)
BashOutput(bash_id="...")  # Check task completion

# 4. Review implementations as they complete
Read("detectrax/models/utils/box_coder.py")  # Review generated code
Read("tests/test_box_coder.py")  # Review generated tests

# 5. Validate
Bash("uv run pytest tests/test_box_coder.py -v")  # Run tests

# 6. Commit if valid
Bash("git add detectrax/models/utils/box_coder.py tests/test_box_coder.py")
Bash('git commit -m "feat: Implement box encoding/decoding (Task 1.2)"')

# 7. Repeat for all completed tasks
# 8. Update ARCHITECTURE_TASKS.md progress tracking
```

### Task Assignment Strategy

**Maximum Parallelism:**
- Identify all tasks with satisfied dependencies
- Launch ALL independent tasks simultaneously
- Example: Sprint 1 tasks 1.2, 1.3, 1.4 have no inter-dependencies → launch all 3 in parallel

**Dependency Management:**
- Sprint 2 (RPN) depends on Sprint 1 (utils) → wait for Sprint 1 completion
- Within a sprint, identify independent sub-tasks and parallelize
- Use ARCHITECTURE_TASKS.md dependency graph to determine launch order

**Resource Considerations:**
- Codex can handle multiple parallel tasks efficiently
- Typical batch size: 3-5 tasks per sprint
- For large sprints, launch in waves (e.g., 4 tasks, review, next 4 tasks)

### Failure Handling

If a task fails or produces incorrect implementation:

1. **Analyze**: Review Codex output logs in `/tmp/task_*_output.json`
2. **Diagnose**: Identify if issue is:
   - Incomplete specification (fix task JSON, rerun)
   - Codex misunderstanding (rephrase prompt, rerun)
   - Genuine edge case (add to requirements, rerun)
3. **Reassign**: Update task JSON and rerun Codex with clarifications
4. **Iterate**: Repeat until implementation passes all checks

**Never implement directly** - always delegate back to Codex with improved specifications.

## Resources

- JAX Documentation: https://jax.readthedocs.io/
- Flax Documentation: https://flax.readthedocs.io/
- Scenic Repository: https://github.com/google-research/scenic
- visdet Reference: `/home/georgepearse/core/machine_learning/packages/visdet`
- zen-mcp-server clink docs: https://github.com/BeehiveInnovations/zen-mcp-server/blob/main/docs/tools/clink.md

---

*For general machine learning guidelines, see the parent machine_learning/CLAUDE.md*
*For general repository guidelines, see the root CLAUDE.md*
