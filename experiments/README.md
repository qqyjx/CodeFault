# Experiments

This directory contains all experiment-related code and results for the AIOps paper.

## Directory Structure

```
experiments/
├── src/              # Source code for methods
│   ├── models/       # Model implementations
│   ├── data/         # Data processing
│   ├── utils/        # Utility functions
│   └── baselines/    # Baseline implementations
├── scripts/          # Running scripts
│   ├── train.sh      # Training script
│   ├── eval.sh       # Evaluation script
│   └── ablation.sh   # Ablation study script
├── configs/          # Configuration files
│   ├── base.yaml     # Base configuration
│   └── experiments/  # Experiment-specific configs
└── logs/             # Training logs

# Note: Results are stored in the root-level results/ directory
```

## Environment Setup

### On A800 Server

```bash
# Create conda environment
conda create -n aiops python=3.10
conda activate aiops

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

## Running Experiments

### Training

```bash
# Train main model
bash scripts/train.sh --config configs/base.yaml

# Train with specific GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train.sh --config configs/base.yaml
```

### Evaluation

```bash
# Evaluate trained model
bash scripts/eval.sh --checkpoint path/to/checkpoint.pt
```

### Ablation Studies

```bash
# Run ablation studies
bash scripts/ablation.sh
```

## Results Organization

All results should be saved in structured format:
- Use consistent naming: `{method}_{dataset}_{timestamp}/`
- Save both raw results and processed metrics
- Include configuration used for reproducibility

## Logging

Use `wandb` or `tensorboard` for experiment tracking:

```python
import wandb
wandb.init(project="aiops-paper", name="experiment-name")
```
