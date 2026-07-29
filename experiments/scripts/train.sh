#!/bin/bash
# Training Script for AIOps Model
# Usage: bash scripts/train.sh --config configs/base.yaml

set -e

# Default values
CONFIG="configs/base.yaml"
GPUS="0,1,2,3"
OUTPUT_DIR="./outputs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
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

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=$GPUS

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# Copy config for reproducibility
cp "$CONFIG" "$RUN_DIR/config.yaml"

echo "=========================================="
echo "Starting Training"
echo "Config: $CONFIG"
echo "GPUs: $GPUS"
echo "Output: $RUN_DIR"
echo "=========================================="

# Run training
python src/train.py \
    --config "$CONFIG" \
    --output_dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train.log"

echo "Training completed. Results saved to $RUN_DIR"
