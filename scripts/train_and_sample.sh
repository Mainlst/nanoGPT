#!/bin/bash
set -e

# Configuration
INTERVAL=${1:-1000} # Default interval: 1000 steps
MAX_ITERS=${2:-10000} # Default max iters: 10000
DATASET="japanese_novel"
OUT_DIR="out-shakespeare-char"
INIT_FROM="scratch"

mkdir -p $OUT_DIR

echo "Starting training loop..."
echo "Interval: $INTERVAL"
echo "Max Iters: $MAX_ITERS"

# Loop from INTERVAL to MAX_ITERS with step INTERVAL
for (( i=$INTERVAL; i<=$MAX_ITERS; i+=$INTERVAL )); do
    echo "========================================================"
    echo "Step 1: Training up to iter $i..."
    
    # Run training
    uv run train.py config/train_shakespeare_char.py \
        --dataset=$DATASET \
        --device=cuda \
        --compile=True \
        --block_size=128 \
        --batch_size=32 \
        --n_layer=4 \
        --n_head=4 \
        --n_embd=256 \
        --max_iters=$i \
        --learning_rate=1e-3 \
        --vocab_source=gpt2 \
        --out_dir=$OUT_DIR \
        --init_from=$INIT_FROM

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
