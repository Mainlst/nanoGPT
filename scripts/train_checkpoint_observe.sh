#!/usr/bin/env bash
set -euo pipefail

# Train in chunks, archive checkpoints, and save samples to observe output drift.
# Usage:
#   bash scripts/train_checkpoint_observe.sh [INTERVAL] [MAX_ITERS]
# Example:
#   bash scripts/train_checkpoint_observe.sh 500 5000
#   WANDB_LOG=false bash scripts/train_checkpoint_observe.sh 500 5000
#   WANDB_SINGLE_RUN=true bash scripts/train_checkpoint_observe.sh 500 5000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/preset_loader.sh"

INTERVAL="${1:-500}"
MAX_ITERS="${2:-5000}"

CONFIG="${CONFIG:-config/train_shakespeare_char.py}"
OUT_DIR="${OUT_DIR:-out-checkpoint-observe}"
DEVICE="${DEVICE:-cuda}"
INIT_FROM="${INIT_FROM:-scratch}"
DATASET="${DATASET:-}"
PRESET="${PRESET:-auto}"
PRESET_FILE="${PRESET_FILE:-config/presets/rtx3070_gpu_presets.json}"
SAVE_CHECKPOINT_ARCHIVE="${SAVE_CHECKPOINT_ARCHIVE:-true}"
WANDB_LOG="${WANDB_LOG:-}"
WANDB_SINGLE_RUN="${WANDB_SINGLE_RUN:-true}"
WANDB_SHARED_RUN_ID="${WANDB_SHARED_RUN_ID:-}"

EVAL_ITERS="${EVAL_ITERS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-1000}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-200}"
SEED="${SEED:-1337}"
START="${START:-\n}"

# Optional training overrides.
DTYPE="${DTYPE:-}"
COMPILE="${COMPILE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
N_LAYER="${N_LAYER:-}"
N_HEAD="${N_HEAD:-}"
N_EMBD="${N_EMBD:-}"

AUTO_PRESET=""
if [[ "$PRESET" == "auto" && "$CONFIG" == "config/train_gpt2.py" ]]; then
  AUTO_PRESET="rtx_3070_8gb_train_gpt2"
fi

SELECTED_PRESET="$PRESET"
if [[ "$PRESET" == "auto" ]]; then
  SELECTED_PRESET="$AUTO_PRESET"
fi

if [[ -n "$SELECTED_PRESET" ]]; then
  load_json_preset_if_unset "$PRESET_FILE" "$SELECTED_PRESET"
fi

CHECKPOINT_ARCHIVE_DIR="$OUT_DIR/checkpoints"
SAMPLE_DIR="$OUT_DIR/samples"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed."
  exit 1
fi

if [[ "$INTERVAL" -le 0 || "$MAX_ITERS" -le 0 || "$INTERVAL" -gt "$MAX_ITERS" ]]; then
  echo "Error: INTERVAL and MAX_ITERS must be positive, and INTERVAL <= MAX_ITERS."
  exit 1
fi

if [[ -n "$WANDB_LOG" && "$WANDB_LOG" != "true" && "$WANDB_LOG" != "false" ]]; then
  echo "Error: WANDB_LOG must be 'true' or 'false' when set."
  exit 1
fi
if [[ "$WANDB_SINGLE_RUN" != "true" && "$WANDB_SINGLE_RUN" != "false" ]]; then
  echo "Error: WANDB_SINGLE_RUN must be 'true' or 'false'."
  exit 1
fi

if [[ "$SAVE_CHECKPOINT_ARCHIVE" == "true" ]]; then
  mkdir -p "$CHECKPOINT_ARCHIVE_DIR"
fi
mkdir -p "$SAMPLE_DIR"

echo "Project root : $ROOT_DIR"
echo "Config       : $CONFIG"
echo "Output dir   : $OUT_DIR"
echo "Device       : $DEVICE"
echo "Interval     : $INTERVAL"
echo "Max iters    : $MAX_ITERS"
echo "Archive ckpt : $SAVE_CHECKPOINT_ARCHIVE"
if [[ -n "$WANDB_LOG" ]]; then
  echo "Wandb log    : $WANDB_LOG (override)"
fi
echo "Wandb single : $WANDB_SINGLE_RUN"
if [[ -n "$SELECTED_PRESET" ]]; then
  echo "Preset       : $SELECTED_PRESET ($PRESET_FILE)"
fi
echo

if [[ "$WANDB_SINGLE_RUN" == "true" ]]; then
  if [[ -z "$WANDB_SHARED_RUN_ID" ]]; then
    WANDB_SHARED_RUN_ID="$(uv run python - <<'PY'
import uuid
print(uuid.uuid4().hex[:12])
PY
)"
  fi
  echo "Wandb run id : $WANDB_SHARED_RUN_ID"
  echo
fi

for ((target=INTERVAL; target<=MAX_ITERS; target+=INTERVAL)); do
  echo "============================================================"
  echo "Training to max_iters=$target (init_from=$INIT_FROM)"

  train_args=(
    "$CONFIG"
    --out_dir="$OUT_DIR"
    --init_from="$INIT_FROM"
    --device="$DEVICE"
    --max_iters="$target"
    --eval_interval="$INTERVAL"
    --eval_iters="$EVAL_ITERS"
    --always_save_checkpoint=True
  )

  if [[ -n "$DATASET" ]]; then
    train_args+=(--dataset="$DATASET")
  fi

  if [[ -n "$DTYPE" ]]; then
    train_args+=(--dtype="$DTYPE")
  fi
  if [[ -n "$COMPILE" ]]; then
    train_args+=(--compile="$COMPILE")
  fi
  if [[ -n "$BATCH_SIZE" ]]; then
    train_args+=(--batch_size="$BATCH_SIZE")
  fi
  if [[ -n "$BLOCK_SIZE" ]]; then
    train_args+=(--block_size="$BLOCK_SIZE")
  fi
  if [[ -n "$GRAD_ACCUM" ]]; then
    train_args+=(--gradient_accumulation_steps="$GRAD_ACCUM")
  fi
  if [[ -n "$N_LAYER" ]]; then
    train_args+=(--n_layer="$N_LAYER")
  fi
  if [[ -n "$N_HEAD" ]]; then
    train_args+=(--n_head="$N_HEAD")
  fi
  if [[ -n "$N_EMBD" ]]; then
    train_args+=(--n_embd="$N_EMBD")
  fi
  if [[ -n "$WANDB_LOG" ]]; then
    train_args+=(--wandb_log="$WANDB_LOG")
  fi

  if [[ "$WANDB_SINGLE_RUN" == "true" ]]; then
    WANDB_RUN_ID="$WANDB_SHARED_RUN_ID" WANDB_RESUME=allow uv run python train.py "${train_args[@]}"
  else
    uv run python train.py "${train_args[@]}"
  fi

  ckpt_path="$OUT_DIR/ckpt.pt"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "Error: checkpoint not found at $ckpt_path"
    exit 1
  fi

  ckpt_copy="$CHECKPOINT_ARCHIVE_DIR/ckpt_iter_${target}.pt"
  if [[ "$SAVE_CHECKPOINT_ARCHIVE" == "true" ]]; then
    cp "$ckpt_path" "$ckpt_copy"
    echo "Archived checkpoint: $ckpt_copy"
  else
    echo "Skipped checkpoint archive (SAVE_CHECKPOINT_ARCHIVE=$SAVE_CHECKPOINT_ARCHIVE)"
  fi

  sample_file="$SAMPLE_DIR/sample_iter_${target}.txt"
  uv run python sample.py \
    --out_dir="$OUT_DIR" \
    --init_from=resume \
    --device="$DEVICE" \
    --seed="$SEED" \
    --start="$START" \
    --num_samples="$NUM_SAMPLES" \
    --max_new_tokens="$SAMPLE_TOKENS" \
    --temperature="$TEMPERATURE" \
    --top_k="$TOP_K" \
    --save_to="$sample_file" >/dev/null

  echo "Saved sample      : $sample_file"
  INIT_FROM="resume"
done

echo
echo "Done."
if [[ "$SAVE_CHECKPOINT_ARCHIVE" == "true" ]]; then
  echo "Checkpoint archive: $CHECKPOINT_ARCHIVE_DIR"
else
  echo "Checkpoint archive: skipped"
fi
echo "Samples           : $SAMPLE_DIR"
