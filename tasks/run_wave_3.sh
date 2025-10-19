#!/bin/bash
# Wave 3: Sprint 5-7 (Losses + Integration + Training + Eval) - 9 tasks

cd /home/georgepearse/FlaxMaskRCNN
TASK_DIR="/home/georgepearse/FlaxMaskRCNN/tasks"
OUTPUT_DIR="/tmp"

echo "=== Wave 3: Launching 9 tasks in parallel ==="
echo "Sprint 5 (Losses): 3 tasks"
echo "Sprint 6 (Integration): 1 task"
echo "Sprint 7 (Training/Eval): 5 tasks"
echo ""

# Sprint 5-7 tasks
for task_file in "$TASK_DIR"/sprint_{5,6,7}/*.json; do
  task_name=$(basename "$task_file" .json)
  (
    echo "[$(date)] $task_name starting..."
    cat "$task_file" | codex exec --json --dangerously-bypass-approvals-and-sandbox > "$OUTPUT_DIR/wave3_${task_name}.json" 2>&1
    echo "[$(date)] $task_name complete"
  ) &
done

echo "All Wave 3 tasks launched (9 tasks)"
echo ""
wait
echo "=== Wave 3 Complete ==="
