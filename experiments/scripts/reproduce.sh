#!/bin/bash
# CodeFault Reproducibility Script
# This script reproduces all experiments from the paper

set -e

echo "=============================================="
echo "CodeFault Experiment Reproduction"
echo "=============================================="

# Check Python version
python3 --version

# Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt

# Generate synthetic data
echo ""
echo "Step 2: Generating synthetic data..."
cd experiments
python scripts/generate_synthetic_data.py \
    --n-bugs 800 \
    --n-normal 400 \
    --difficulty hard \
    --output-dir ../data/processed/codefault

# Run experiments (single run for quick verification)
echo ""
echo "Step 3: Running experiments (quick mode)..."
python run_all_experiments.py --rqs rq1 rq2 rq3 rq4 --num-runs 1

# For full paper results with statistical significance, use:
# python run_all_experiments.py --rqs rq1 rq2 rq3 rq4 --num-runs 5

# Generate figures
echo ""
echo "Step 4: Generating figures..."
python generate_figures.py

echo ""
echo "=============================================="
echo "Reproduction complete!"
echo "Results saved to: results/"
echo "Figures saved to: paper/figures/"
echo "=============================================="
