#!/usr/bin/env bash
set -euo pipefail

# nanoGPT OpenWebText training script for a single RTX 3070 (8GB).
# Usage:
#   bash scripts/train_openwebtext_3070.sh
# Optional env overrides:
#   MAX_ITERS=100000 OUT_DIR=out-openwebtext-3070 bash scripts/train_openwebtext_3070.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/preset_loader.sh"

DATA_DIR="data/openwebtext"
TRAIN_BIN="$DATA_DIR/train.bin"
VAL_BIN="$DATA_DIR/val.bin"

OUT_DIR="${OUT_DIR:-out-openwebtext-3070}"
MAX_ITERS="${MAX_ITERS:-50000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-200}"
WANDB_LOG="${WANDB_LOG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-owt}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-gpt2-3070}"
PRESET="${PRESET:-auto}"
PRESET_FILE="${PRESET_FILE:-config/presets/rtx3070_gpu_presets.json}"

DTYPE="${DTYPE:-}"
COMPILE="${COMPILE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
N_LAYER="${N_LAYER:-}"
N_HEAD="${N_HEAD:-}"
N_EMBD="${N_EMBD:-}"

AUTO_PRESET=""
if [[ "$PRESET" == "auto" ]]; then
  AUTO_PRESET="rtx_3070_8gb_train_gpt2"
fi

SELECTED_PRESET="$PRESET"
if [[ "$PRESET" == "auto" ]]; then
  SELECTED_PRESET="$AUTO_PRESET"
fi

if [[ -n "$SELECTED_PRESET" ]]; then
  load_json_preset_if_unset "$PRESET_FILE" "$SELECTED_PRESET"
fi

echo "Project root : $ROOT_DIR"
echo "Output dir   : $OUT_DIR"
echo "Max iters    : $MAX_ITERS"
echo "W&B logging  : $WANDB_LOG (project=$WANDB_PROJECT, run=$WANDB_RUN_NAME)"
if [[ -n "$SELECTED_PRESET" ]]; then
  echo "Preset       : $SELECTED_PRESET ($PRESET_FILE)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not installed. Install uv first, then retry."
  exit 1
fi

if [[ ! -f "$TRAIN_BIN" || ! -f "$VAL_BIN" ]]; then
  echo "OpenWebText bin files not found. Running prepare step..."
  uv run python "$DATA_DIR/prepare.py"
fi

mkdir -p "$OUT_DIR"

echo "Starting training..."

train_args=(
  config/train_gpt2.py
  --dataset=openwebtext
  --device=cuda
  --eval_interval="$EVAL_INTERVAL"
  --eval_iters=50
  --log_interval="$LOG_INTERVAL"
  --max_iters="$MAX_ITERS"
  --lr_decay_iters="$MAX_ITERS"
  --out_dir="$OUT_DIR"
  --wandb_log="$WANDB_LOG"
  --wandb_project="$WANDB_PROJECT"
  --wandb_run_name="$WANDB_RUN_NAME"
)

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

uv run python train.py "${train_args[@]}"

echo "Training complete. Generating a sample..."
uv run python sample.py \
  --out_dir="$OUT_DIR" \
  --device=cuda \
  --max_new_tokens="$SAMPLE_TOKENS" \
  > "$OUT_DIR/sample_latest.txt"

echo "Done."
echo "Sample written to: $OUT_DIR/sample_latest.txt"
