#!/usr/bin/env bash
set -euo pipefail

# nanoGPT OpenWebText training script for a single RTX 3070 (8GB).
# Usage:
#   bash scripts/train_openwebtext_3070.sh
# Optional env overrides:
#   MAX_ITERS=100000 OUT_DIR=out-openwebtext-3070 bash scripts/train_openwebtext_3070.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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

echo "Project root : $ROOT_DIR"
echo "Output dir   : $OUT_DIR"
echo "Max iters    : $MAX_ITERS"
echo "W&B logging  : $WANDB_LOG (project=$WANDB_PROJECT, run=$WANDB_RUN_NAME)"

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
uv run python train.py config/train_gpt2.py \
  --dataset=openwebtext \
  --device=cuda \
  --dtype=float16 \
  --compile=False \
  --batch_size=2 \
  --block_size=512 \
  --gradient_accumulation_steps=16 \
  --n_layer=8 \
  --n_head=8 \
  --n_embd=512 \
  --eval_interval="$EVAL_INTERVAL" \
  --eval_iters=50 \
  --log_interval="$LOG_INTERVAL" \
  --max_iters="$MAX_ITERS" \
  --lr_decay_iters="$MAX_ITERS" \
  --out_dir="$OUT_DIR" \
  --wandb_log="$WANDB_LOG" \
  --wandb_project="$WANDB_PROJECT" \
  --wandb_run_name="$WANDB_RUN_NAME"

echo "Training complete. Generating a sample..."
uv run python sample.py \
  --out_dir="$OUT_DIR" \
  --device=cuda \
  --max_new_tokens="$SAMPLE_TOKENS" \
  > "$OUT_DIR/sample_latest.txt"

echo "Done."
echo "Sample written to: $OUT_DIR/sample_latest.txt"
