# CodeFault: Predicting Fault Risks from Code Changes via Multi-modal Learning

Replication package for the ASE 2026 paper **"CodeFault: Predicting Fault Risks from Code Changes via Multi-modal Learning"** (paper #759).

A long-term archived copy of this repository (including all data) is deposited on Zenodo: https://doi.org/10.5281/zenodo.21698209 (this DOI always resolves to the latest version).

## Contents

```
├── experiments/
│   ├── src/                  # CodeFault model, training and evaluation code
│   │   └── models/           # Model implementations (HFSA, SGCA, fusion)
│   ├── baselines/            # All 15 baseline implementations + 2 late-fusion controls
│   ├── scripts/              # Dataset download / preprocessing / mining scripts
│   ├── configs/              # Training configuration (base.yaml)
│   ├── run_all_experiments.py        # RQ1–RQ4 driver
│   ├── run_realworld_validation.py   # CI/CD deployment validation (RQ7)
│   ├── run_sensitivity_analysis.py   # Hyperparameter sensitivity
│   ├── run_efficiency_analysis.py    # Latency / cost analysis
│   └── run_motivation_study.py       # Motivating study
├── data/processed/codefault/
│   ├── train.json / val.json / test.json   # Chronological 70/15/15 split used in the paper
│   ├── bugsinpy.json / defects4j.json / manysstubs4j.json  # Processed source datasets
│   ├── normal_commits.json                 # Mined benign commits
│   └── metadata.json                       # Dataset statistics and fault-type taxonomy
├── results/                  # JSON results backing every table and figure,
│                             # incl. rq7_deployment.json (real CI/CD aggregate results)
└── requirements.txt
```

Raw third-party datasets (Defects4J, BugsInPy, ManySStuBs4J, Loghub) are not
redistributed here; `experiments/scripts/download_datasets.py` fetches them and
`experiments/scripts/process_*.py` reproduce the processed files above.

## Setup

```bash
conda create -n codefault python=3.10 -y
conda activate codefault
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Reproducing the paper

```bash
# Main results (RQ1–RQ4): training + evaluation on the provided splits
python experiments/run_all_experiments.py --config experiments/configs/base.yaml

# Real-world CI/CD validation (RQ7)
python experiments/run_realworld_validation.py

# Sensitivity / efficiency / motivation analyses
python experiments/run_sensitivity_analysis.py
python experiments/run_efficiency_analysis.py
python experiments/run_motivation_study.py
```

Training in the paper used four NVIDIA A800 80GB GPUs (batch size 32, lr 2e-5,
10 epochs, AdamW); reported numbers are means over 10 fixed-seed runs.
Pre-computed outputs for every table and figure are already in `results/`.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{xiao2026codefault,
  series = {ASE ’26},
  title = {CodeFault: Predicting Fault Risks from Code Changes via Multi-modal Learning},
  url = {http://dx.doi.org/10.1145/3832783.3837462},
  DOI = {10.1145/3832783.3837462},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Automated Software Engineering},
  publisher = {ACM},
  author = {Xiao, Yifan and Li, Shijie and Huang, Yu},
  year = {2026},
  month = Oct,
  pages = {2101–2113},
  collection = {ASE ’26},
}
```

### Artifact

The replication package is archived on Zenodo:

```bibtex
@misc{xiao2026codefaultartifact,
  doi = {10.5281/ZENODO.21700762},
  url = {https://zenodo.org/doi/10.5281/zenodo.21700762},
  author = {Xiao, Yifan and Li, Shijie and Huang, Yu},
  title = {CodeFault: Predicting Fault Risks from Code Changes via Multi-modal Learning — Replication Package (ASE 2026)},
  publisher = {Zenodo},
  year = {2026},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

MIT — see [LICENSE](LICENSE).
