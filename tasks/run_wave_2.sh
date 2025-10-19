#!/bin/bash
# Wave 2: Sprint 3-4 (Detection Head + Mask Head) - 7 tasks in parallel

cd /home/georgepearse/FlaxMaskRCNN
TASK_DIR="/home/georgepearse/FlaxMaskRCNN/tasks"
OUTPUT_DIR="/tmp"

echo "=== Wave 2: Launching 7 tasks in parallel ==="
echo "Sprint 3 (Detection Head): 4 tasks"
echo "Sprint 4 (Mask Head): 3 tasks"
echo ""

# Sprint 3 tasks
for task_file in "$TASK_DIR"/sprint_3/*.json; do
  task_name=$(basename "$task_file" .json)
  (
    echo "[$(date)] $task_name starting..."
    cat "$task_file" | codex exec --json --dangerously-bypass-approvals-and-sandbox > "$OUTPUT_DIR/wave2_${task_name}.json" 2>&1
    echo "[$(date)] $task_name complete"
  ) &
done

# Sprint 4 tasks
for task_file in "$TASK_DIR"/sprint_4/*.json; do
  task_name=$(basename "$task_file" .json)
  (
    echo "[$(date)] $task_name starting..."
    cat "$task_file" | codex exec --json --dangerously-bypass-approvals-and-sandbox > "$OUTPUT_DIR/wave2_${task_name}.json" 2>&1
    echo "[$(date)] $task_name complete"
  ) &
done

echo "All Wave 2 tasks launched (7 tasks)"
echo ""
wait
echo "=== Wave 2 Complete ==="
