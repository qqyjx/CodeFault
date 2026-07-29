#!/usr/bin/env python3
"""
RQ9: Real-World Validation Study

Tests CodeFault and baselines on real CI/CD code-log pairs collected
from GitHub Actions, addressing the construct validity concern about
synthetic code-log pairing.

Two modes:
  1. Zero-shot transfer: model trained on synthetic data, tested on real data
  2. Domain gap analysis: quantifies representation distance between
     synthetic and real code-log embeddings

Usage:
    python run_realworld_validation.py
    python run_realworld_validation.py --mode zero-shot
    python run_realworld_validation.py --mode domain-gap

Output:
    results/rq9_realworld_validation.json
"""

import os
import sys
import json
import time
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
)
from scipy.spatial.distance import cosine as cosine_distance

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.models.codefault import (
    CodeFault as FullCodeFault,
    CodeChangeEncoder, FaultContextEncoder,
    FaultSpectrumAlignment, TemporalCausalFusion,
    CrossModalFusion, ContrastiveLoss
)

from run_all_experiments import (
    NumpyEncoder,
    ExperimentCodeFault,
    FaultPredictionDataset,
    train_model,
)


class RealCIDataset(Dataset):
    """Dataset for real CI/CD code-log pairs."""

    def __init__(self, data_path: Path, max_code_len: int = 512, max_log_len: int = 128):
        with open(data_path) as f:
            self.samples = json.load(f)
        self.max_code_len = max_code_len
        self.max_log_len = max_log_len
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from real CI data."""
        word_counts = {}
        for sample in self.samples:
            diff = sample["code_change"]["diff"]
            for token in diff.split():
                word_counts[token] = word_counts.get(token, 0) + 1
            for msg in sample["log_context"]["messages"]:
                for token in msg.split():
                    word_counts[token] = word_counts.get(token, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        for word, _ in sorted_words[:9998]:
            self.word2idx[word] = len(self.word2idx)

    def _tokenize(self, text: str, max_len: int) -> torch.Tensor:
        tokens = text.split()[:max_len]
        indices = [self.word2idx.get(t, 1) for t in tokens]
        while len(indices) < max_len:
            indices.append(0)
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        diff = sample["code_change"]["diff"]
        code_ids = self._tokenize(diff, self.max_code_len)
        log_text = " ".join(sample["log_context"]["messages"])
        log_ids = self._tokenize(log_text, self.max_log_len)
        label = torch.tensor(sample["label"], dtype=torch.float)
        return code_ids, log_ids, label


def compute_all_metrics(y_true, y_pred, y_prob):
    """Compute full metric suite."""
    metrics = {
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    n = len(y_true)
    top_k = max(1, int(n * 0.2))
    top_indices = np.argsort(y_prob)[-top_k:]
    metrics["p_at_20"] = float(np.mean([y_true[i] for i in top_indices]))
    return metrics


def run_model_on_real_data(
    model: nn.Module,
    real_dataset: RealCIDataset,
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    """Run a trained model on real CI/CD data and compute metrics."""
    model.train(False)
    loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for code_ids, log_ids, labels in loader:
            code_ids = code_ids.to(device)
            log_ids = log_ids.to(device)
            outputs = model(code_ids, log_ids)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.numpy().flatten().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_all_metrics(y_true, y_pred, y_prob)


def extract_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract code and log embeddings from a model."""
    model.train(False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    code_embeddings = []
    log_embeddings = []

    with torch.no_grad():
        for code_ids, log_ids, _ in loader:
            code_ids = code_ids.to(device)
            log_ids = log_ids.to(device)
            if hasattr(model, 'code_encoder') and hasattr(model, 'log_encoder'):
                code_emb = model.code_encoder(code_ids)
                log_emb = model.log_encoder(log_ids)
            else:
                outputs = model(code_ids, log_ids)
                code_emb = outputs.get("code_embedding", torch.zeros(code_ids.size(0), 256))
                log_emb = outputs.get("log_embedding", torch.zeros(log_ids.size(0), 256))
            code_embeddings.append(code_emb.cpu().numpy())
            log_embeddings.append(log_emb.cpu().numpy())

    return np.concatenate(code_embeddings, axis=0), np.concatenate(log_embeddings, axis=0)


def compute_domain_gap(
    syn_code: np.ndarray, syn_log: np.ndarray,
    real_code: np.ndarray, real_log: np.ndarray,
) -> dict:
    """Compute domain gap metrics between synthetic and real embeddings."""
    code_gap = float(cosine_distance(syn_code.mean(0), real_code.mean(0)))
    log_gap = float(cosine_distance(syn_log.mean(0), real_log.mean(0)))

    def mmd_rbf(X, Y, gamma=1.0):
        n = min(len(X), len(Y), 500)
        Xi = X[np.random.choice(len(X), n, replace=False)]
        Yi = Y[np.random.choice(len(Y), n, replace=False)]
        XX = np.dot(Xi, Xi.T)
        YY = np.dot(Yi, Yi.T)
        XY = np.dot(Xi, Yi.T)
        Xn = np.diagonal(XX)
        Yn = np.diagonal(YY)
        K_XX = np.exp(-gamma * (Xn[:, None] + Xn[None, :] - 2 * XX))
        K_YY = np.exp(-gamma * (Yn[:, None] + Yn[None, :] - 2 * YY))
        K_XY = np.exp(-gamma * (Xn[:, None] + Yn[None, :] - 2 * XY))
        return float(max(0, K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()))

    return {
        "code_cosine_gap": code_gap,
        "log_cosine_gap": log_gap,
        "code_mmd": mmd_rbf(syn_code, real_code),
        "log_mmd": mmd_rbf(syn_log, real_log),
        "mean_gap": (code_gap + log_gap) / 2,
    }


def run_zero_shot_transfer(
    real_data_path: Path,
    synthetic_data_path: Path,
    results_dir: Path,
    seeds: list = None,
    device: torch.device = None,
) -> dict:
    """Train on synthetic data, test on real CI/CD data."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("RQ9 Part 1: Zero-Shot Transfer")
    print("=" * 60)

    real_dataset = RealCIDataset(real_data_path)
    print(f"Real CI/CD dataset: {len(real_dataset)} samples")

    model_configs = {
        "CodeFault": {"ablation": None},
        "CodeFault-NoLog": {"ablation": "no_log"},
        "CodeFault-NoCross": {"ablation": "no_cross"},
    }

    all_results = {}
    for model_name, cfg in model_configs.items():
        print(f"\nTesting {model_name}...")
        seed_results = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            kwargs = {
                "hidden_dim": 256, "num_heads": 8, "num_layers": 3,
                "code_vocab_size": 10000, "log_vocab_size": 10000,
            }
            if cfg["ablation"]:
                kwargs["ablation"] = cfg["ablation"]

            model = ExperimentCodeFault(**kwargs).to(device)

            if synthetic_data_path.exists():
                syn_dataset = FaultPredictionDataset(str(synthetic_data_path))
                model = train_model(model, syn_dataset, epochs=10, lr=2e-5, device=device, verbose=False)

            metrics = run_model_on_real_data(model, real_dataset, device)
            seed_results.append(metrics)
            print(f"  Seed {seed}: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

        agg = {}
        for key in seed_results[0]:
            values = [r[key] for r in seed_results]
            agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
        all_results[model_name] = agg

    return all_results


def run_domain_gap_analysis(
    real_data_path: Path,
    synthetic_data_path: Path,
    results_dir: Path,
    device: torch.device = None,
) -> dict:
    """Quantify representation distance between synthetic and real data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("RQ9 Part 2: Domain Gap Analysis")
    print("=" * 60)

    real_dataset = RealCIDataset(real_data_path)
    if not synthetic_data_path.exists():
        print("WARNING: Synthetic data not found. Skipping.")
        return {}

    syn_dataset = FaultPredictionDataset(str(synthetic_data_path))
    model = ExperimentCodeFault(
        hidden_dim=256, num_heads=8, num_layers=3,
        code_vocab_size=10000, log_vocab_size=10000,
    ).to(device)
    model = train_model(model, syn_dataset, epochs=10, lr=2e-5, device=device)

    print("Extracting embeddings...")
    syn_code, syn_log = extract_embeddings(model, syn_dataset, device)
    real_code, real_log = extract_embeddings(model, real_dataset, device)
    print(f"  Synthetic: code={syn_code.shape}, log={syn_log.shape}")
    print(f"  Real:      code={real_code.shape}, log={real_log.shape}")

    gap = compute_domain_gap(syn_code, syn_log, real_code, real_log)
    print(f"\nDomain Gap:")
    for k, v in gap.items():
        print(f"  {k}: {v:.4f}")
    return gap


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RQ9: Real-World Validation Study")
    parser.add_argument("--mode", choices=["full", "zero-shot", "domain-gap"], default="full")
    parser.add_argument("--real-data", type=Path,
                        default=SCRIPT_DIR / "data" / "real_ci" / "real_ci_eval.json")
    parser.add_argument("--synthetic-data", type=Path,
                        default=SCRIPT_DIR / "data" / "processed" / "codefault" / "train.json")
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_DIR.parent / "results" / "rq9_realworld_validation.json")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144])
    args = parser.parse_args()

    if not args.real_data.exists():
        print(f"ERROR: Real CI/CD data not found at {args.real_data}")
        print("Run collect_github_actions.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {
        "_meta": {
            "experiment": "RQ9: Real-World Validation Study",
            "real_data_path": str(args.real_data),
            "synthetic_data_path": str(args.synthetic_data),
            "timestamp": datetime.now().isoformat(),
            "seeds": args.seeds,
        }
    }

    results_dir = args.output.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("full", "zero-shot"):
        results["zero_shot_transfer"] = run_zero_shot_transfer(
            args.real_data, args.synthetic_data, results_dir,
            seeds=args.seeds, device=device,
        )

    if args.mode in ("full", "domain-gap"):
        results["domain_gap"] = run_domain_gap_analysis(
            args.real_data, args.synthetic_data, results_dir, device=device,
        )

    # Transfer efficiency vs. synthetic benchmark
    if "zero_shot_transfer" in results:
        syn_path = results_dir / "rq1_model_comparison.json"
        if syn_path.exists():
            with open(syn_path) as f:
                syn_data = json.load(f)
            if "all_runs" in syn_data and syn_data["all_runs"]:
                syn_cf = syn_data["all_runs"][0].get("CodeFault", {})
                real_cf = results["zero_shot_transfer"].get("CodeFault", {})
                if syn_cf and real_cf:
                    efficiency = {}
                    for key in ["auc_roc", "auc_pr", "f1", "mcc"]:
                        sv = syn_cf.get(key, 0)
                        rv = real_cf.get(key, {}).get("mean", 0)
                        if sv > 0:
                            efficiency[key] = {
                                "synthetic": sv, "real": rv,
                                "retention": rv / sv,
                                "gap": sv - rv,
                            }
                    results["transfer_efficiency"] = efficiency

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
