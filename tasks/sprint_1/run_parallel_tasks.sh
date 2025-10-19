#!/bin/bash
# Run Sprint 1 tasks 1.2-1.4 in parallel via Codex

cd /home/georgepearse/FlaxMaskRCNN

echo "Starting parallel Codex tasks..."

# Task 1.2: Box Encoding/Decoding
(
  echo "Task 1.2: Box Coder starting..."
  cat /tmp/task_1_2_box_coder.json | codex exec --json --dangerously-bypass-approvals-and-sandbox > /tmp/task_1_2_output.json 2>&1
  echo "Task 1.2: Box Coder complete"
) &
PID_1_2=$!

# Task 1.3: NMS
(
  echo "Task 1.3: NMS starting..."
  cat /tmp/task_1_3_nms.json | codex exec --json --dangerously-bypass-approvals-and-sandbox > /tmp/task_1_3_output.json 2>&1
  echo "Task 1.3: NMS complete"
) &
PID_1_3=$!

# Task 1.4: IoU
(
  echo "Task 1.4: IoU starting..."
  cat /tmp/task_1_4_iou.json | codex exec --json --dangerously-bypass-approvals-and-sandbox > /tmp/task_1_4_output.json 2>&1
  echo "Task 1.4: IoU complete"
) &
PID_1_4=$!

echo "All tasks launched in background"
echo "PIDs: Box Coder=$PID_1_2, NMS=$PID_1_3, IoU=$PID_1_4"
echo ""
echo "Monitor with: tail -f /tmp/task_1_*.json"
echo "Wait for completion with: wait"

# Wait for all to complete
wait $PID_1_2
wait $PID_1_3
wait $PID_1_4

echo ""
echo "All Sprint 1 tasks complete!"
echo "Check outputs in /tmp/task_1_*.json"
