#!/usr/bin/env bash
set -e  # exit on error

# ------------------------------
# Usage and argument parsing
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
# Generate timestamp and paths
# ------------------------------
time="$(date +%Y-%m-%d_%H-%M-%S)"
job_name="${time}_act_${name}"
output_dir="outputs/checkpoints/${task}/${job_name}"

# ------------------------------
# Run training
# ------------------------------
echo "Launching training..."
python third_party/lerobot/lerobot/scripts/train.py \
  --config_path="configs/training/act_${task}.json" \
  --job_name="${job_name}" \
  --output_dir="${output_dir}"
