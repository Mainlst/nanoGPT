#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/preset_loader.sh"

# Configuration
INTERVAL=${1:-1000} # Default interval: 1000 steps
MAX_ITERS=${2:-10000} # Default max iters: 10000
DATASET="japanese_novel"
OUT_DIR="out-shakespeare-char"
INIT_FROM="scratch"
PRESET="${PRESET:-}"
PRESET_FILE="${PRESET_FILE:-config/presets/rtx3070_gpu_presets.json}"

DTYPE="${DTYPE:-}"
COMPILE="${COMPILE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
GRAD_ACCUM="${GRAD_ACCUM:-}"
N_LAYER="${N_LAYER:-4}"
N_HEAD="${N_HEAD:-4}"
N_EMBD="${N_EMBD:-256}"

if [[ -n "$PRESET" ]]; then
    load_json_preset_if_unset "$PRESET_FILE" "$PRESET"
fi

mkdir -p $OUT_DIR

echo "Starting training loop..."
echo "Interval: $INTERVAL"
echo "Max Iters: $MAX_ITERS"
if [[ -n "$PRESET" ]]; then
    echo "Preset: $PRESET ($PRESET_FILE)"
fi

# Loop from INTERVAL to MAX_ITERS with step INTERVAL
for (( i=$INTERVAL; i<=$MAX_ITERS; i+=$INTERVAL )); do
    echo "========================================================"
    echo "Step 1: Training up to iter $i..."
    
    # Run training
    train_args=(
        config/train_shakespeare_char.py
        --dataset="$DATASET"
        --device=cuda
        --block_size="$BLOCK_SIZE"
        --n_layer="$N_LAYER"
        --n_head="$N_HEAD"
        --n_embd="$N_EMBD"
        --max_iters="$i"
        --learning_rate=1e-3
        --vocab_source=gpt2
        --out_dir="$OUT_DIR"
        --init_from="$INIT_FROM"
    )

    if [[ -n "$DTYPE" ]]; then
        train_args+=(--dtype="$DTYPE")
    fi
    if [[ -n "$COMPILE" ]]; then
        train_args+=(--compile="$COMPILE")
    else
        train_args+=(--compile=True)
    fi
    if [[ -n "$BATCH_SIZE" ]]; then
        train_args+=(--batch_size="$BATCH_SIZE")
    else
        train_args+=(--batch_size=32)
    fi
    if [[ -n "$GRAD_ACCUM" ]]; then
        train_args+=(--gradient_accumulation_steps="$GRAD_ACCUM")
    fi

    uv run train.py "${train_args[@]}"

    echo "Step 2: Sampling at iter $i..."
    # Run sampling
    uv run sample.py \
        --out_dir=$OUT_DIR \
        --device=cuda \
        --init_from=resume \
        > "$OUT_DIR/sample_$i.txt"
        
    echo "Sample saved to $OUT_DIR/sample_$i.txt"
    
    # Ensure next iteration resumes from checkpoint
    INIT_FROM="resume"
done

echo "Done!"
