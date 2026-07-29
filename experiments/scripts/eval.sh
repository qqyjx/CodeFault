#!/bin/bash
# Evaluation Script for AIOps Model
# Usage: bash scripts/eval.sh --checkpoint path/to/checkpoint.pt

set -e

# Default values
CHECKPOINT=""
CONFIG="configs/base.yaml"
OUTPUT_DIR="./eval_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting Evaluation"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run evaluation
python src/eval.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo "Evaluation completed. Results saved to $OUTPUT_DIR"
