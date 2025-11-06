#!/usr/bin/env bash
set -e  # Exit immediately if any command fails

# ------------------------------
# Usage and arguments
# ------------------------------
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <experiment_name>"
  echo "Allowed tasks: insert_rope | pack_sloth | pusht"
  exit 1
fi

task="$1"
name="$2"

# ------------------------------
# Validate task
# ------------------------------
case "$task" in
  insert_rope|pack_sloth|pusht)
    echo "Task: $task"
    ;;
  *)
    echo "Invalid task: '$task'"
    echo "Allowed tasks: insert_rope | pack_sloth | pusht"
    exit 1
    ;;
esac

# ------------------------------
# Run commands
# ------------------------------
echo "Computing normalization stats... (comment out if already computed)"
uv run third_party/openpi/scripts/compute_norm_stats.py \
  --config-name "pi0_lora_${task}"

echo "Launching training..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run third_party/openpi/scripts/train.py \
  "pi0_lora_${task}" \
  --exp-name="pi0_${name}" \
  --overwrite
