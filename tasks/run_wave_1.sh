#!/bin/bash
# Wave 1: NMS Fix + Sprint 2 RPN (4 tasks in parallel)

cd /home/georgepearse/FlaxMaskRCNN

TASK_DIR="/home/georgepearse/FlaxMaskRCNN/tasks"
OUTPUT_DIR="/tmp"

echo "=== Wave 1: Launching 4 tasks in parallel ==="
echo "1. NMS Fix"
echo "2. RPN Head"
echo "3. RPN Assigner"
echo "4. RPN Proposals"
echo ""

# Task 1.3 Fix: NMS
(
  echo "[$(date)] NMS Fix starting..."
  cat "$TASK_DIR/sprint_1/task_1_3_nms_fix.json" | codex exec --json --dangerously-bypass-approvals-and-sandbox > "$OUTPUT_DIR/wave1_nms_fix.json" 2>&1
  echo "[$(date)] NMS Fix complete"
) &
PID_NMS=$!

# Task 2.1: RPN Head
(
  echo "[$(date)] RPN Head starting..."
  cat "$TASK_DIR/sprint_2/task_2_1_rpn_head.json" | codex exec --json --dangerously-bypass-approvals-and-sandbox > "$OUTPUT_DIR/wave1_rpn_head.json" 2>&1
  echo "[$(date)] RPN Head complete"
) &
PID_RPN_HEAD=$!

# Task 2.2: RPN Assigner
(
  echo "[$(date)] RPN Assigner starting..."
  cat "$TASK_DIR/sprint_2/task_2_2_rpn_assigner.json" | codex exec --json --dangerously-bypass-approvals-and-sandbox > "$OUTPUT_DIR/wave1_rpn_assigner.json" 2>&1
  echo "[$(date)] RPN Assigner complete"
) &
PID_RPN_ASSIGN=$!

# Task 2.3: RPN Proposals
(
  echo "[$(date)] RPN Proposals starting..."
  cat "$TASK_DIR/sprint_2/task_2_3_rpn_proposals.json" | codex exec --json --dangerously-bypass-approvals-and-sandbox > "$OUTPUT_DIR/wave1_rpn_proposals.json" 2>&1
  echo "[$(date)] RPN Proposals complete"
) &
PID_RPN_PROP=$!

echo "All Wave 1 tasks launched"
echo "PIDs: NMS=$PID_NMS, RPN_Head=$PID_RPN_HEAD, RPN_Assigner=$PID_RPN_ASSIGN, RPN_Proposals=$PID_RPN_PROP"
echo ""
echo "Monitor: tail -f $OUTPUT_DIR/wave1_*.json"
echo ""

# Wait for all
wait $PID_NMS
wait $PID_RPN_HEAD
wait $PID_RPN_ASSIGN
wait $PID_RPN_PROP

echo ""
echo "=== Wave 1 Complete ==="
echo "Check outputs in $OUTPUT_DIR/wave1_*.json"
