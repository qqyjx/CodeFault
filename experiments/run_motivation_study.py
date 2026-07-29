#!/usr/bin/env python3
"""
Motivation Study: Modality Complementarity Analysis

Trains code-only and log-only models independently, then analyzes:
1. Error set overlap (Venn diagram data)
2. Per-fault-type complementarity
3. Statistical evidence that multi-modal fusion is necessary

This provides empirical motivation for CodeFault's multi-modal design,
replacing the anecdotal Tomcat example with quantitative evidence.

Usage:
    python run_motivation_study.py
"""

import json
import sys
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from run_all_experiments import (
    ExperimentCodeFault, CodeFaultDataset, CodeOnlyStudent,
    set_seed, train_model, DEVICE, DATA_DIR, RESULTS_DIR, NumpyEncoder
)


def train_log_only_model(vocab_size, train_loader, epochs=15):
    """Train a log-only model (no code encoder)."""
    model = ExperimentCodeFault(
        vocab_size, use_spectrum=False, use_tcf=False, use_contrastive=True
    ).to(DEVICE)

    # Zero out code encoder contribution by replacing code with zeros
    class LogOnlyWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model

        def forward(self, code_ids, code_mask, log_ids, log_mask,
                    labels=None, fault_types=None):
            dummy_code = torch.zeros_like(code_ids)
            dummy_mask = torch.zeros_like(code_mask)
            return self.base(dummy_code, dummy_mask, log_ids, log_mask,
                           labels=labels, fault_types=fault_types)

    wrapper = LogOnlyWrapper(model).to(DEVICE)
    train_model(wrapper, train_loader, train_loader, epochs=epochs, device=DEVICE)
    return wrapper


def get_predictions(model, loader, code_only=False):
    """Get per-sample predictions from a model."""
    model.eval()
    all_probs = []
    all_labels = []
    all_fault_types = []

    with torch.no_grad():
        for batch in loader:
            code_ids = batch['code_input_ids'].to(DEVICE)
            code_mask = batch['code_attention_mask'].to(DEVICE)
            log_ids = batch['log_input_ids'].to(DEVICE)
            log_mask = batch['log_attention_mask'].to(DEVICE)

            if code_only:
                out = model(code_ids, code_mask)
            else:
                out = model(code_ids, code_mask, log_ids, log_mask)

            all_probs.extend(out['probability'].cpu().tolist())
            all_labels.extend(batch['label'].tolist())
            all_fault_types.extend(batch['fault_type'].tolist())

    return np.array(all_probs), np.array(all_labels), np.array(all_fault_types)


def compute_error_sets(probs, labels, threshold=0.5):
    """Compute correct/incorrect prediction sets."""
    preds = (probs >= threshold).astype(int)
    correct = set(np.where(preds == labels)[0])
    incorrect = set(np.where(preds != labels)[0])
    return correct, incorrect


def main():
    print("=" * 60)
    print("Motivation Study: Modality Complementarity Analysis")
    print("=" * 60)

    train_dataset = CodeFaultDataset(DATA_DIR / "train.json")
    test_dataset = CodeFaultDataset(DATA_DIR / "test.json", vocab=train_dataset.vocab)
    vocab_size = len(train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    fault_type_names = [
        'NullRef', 'TypeMismatch', 'Boundary', 'ResLeak',
        'Config', 'APIViol', 'Logic', 'Concurrency', 'Normal'
    ]

    # 1. Train code-only model
    print("\n1. Training code-only model...")
    set_seed(42)
    code_model = CodeOnlyStudent(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(code_model.parameters(), lr=1e-3, weight_decay=0.01)
    for epoch in range(15):
        code_model.train()
        for batch in train_loader:
            code_ids = batch['code_input_ids'].to(DEVICE)
            code_mask = batch['code_attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            out = code_model(code_ids, code_mask)
            loss = F.binary_cross_entropy_with_logits(out['logits'], labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    code_probs, labels, fault_types = get_predictions(code_model, test_loader, code_only=True)

    # 2. Train log-only model
    print("\n2. Training log-only model...")
    set_seed(42)
    log_model = train_log_only_model(vocab_size, train_loader, epochs=15)
    log_probs, _, _ = get_predictions(log_model, test_loader)

    # 3. Train full multi-modal model
    print("\n3. Training full multi-modal model...")
    set_seed(42)
    full_model = ExperimentCodeFault(vocab_size).to(DEVICE)
    train_model(full_model, train_loader, train_loader, epochs=15, device=DEVICE)
    full_probs, _, _ = get_predictions(full_model, test_loader)

    # 4. Compute error sets and Venn diagram data
    print("\n4. Computing modality complementarity...")

    code_correct, code_incorrect = compute_error_sets(code_probs, labels)
    log_correct, log_incorrect = compute_error_sets(log_probs, labels)
    full_correct, full_incorrect = compute_error_sets(full_probs, labels)

    n_total = len(labels)
    code_only_correct = code_correct - log_correct
    log_only_correct = log_correct - code_correct
    both_correct = code_correct & log_correct
    neither_correct = code_incorrect & log_incorrect
    full_recovers = full_correct & neither_correct

    venn_data = {
        'total_samples': n_total,
        'code_only_correct': len(code_only_correct),
        'log_only_correct': len(log_only_correct),
        'both_correct': len(both_correct),
        'neither_correct': len(neither_correct),
        'full_model_recovers': len(full_recovers),
        'complementarity_rate': (len(code_only_correct) + len(log_only_correct)) / n_total,
    }

    print(f"\n  Venn Diagram Data:")
    print(f"    Code-only correct: {venn_data['code_only_correct']} ({venn_data['code_only_correct']/n_total:.1%})")
    print(f"    Log-only correct:  {venn_data['log_only_correct']} ({venn_data['log_only_correct']/n_total:.1%})")
    print(f"    Both correct:      {venn_data['both_correct']} ({venn_data['both_correct']/n_total:.1%})")
    print(f"    Neither correct:   {venn_data['neither_correct']} ({venn_data['neither_correct']/n_total:.1%})")
    print(f"    Complementarity:   {venn_data['complementarity_rate']:.1%}")
    print(f"    Full recovers:     {venn_data['full_model_recovers']}")

    # 5. Per-fault-type complementarity
    print("\n5. Per-fault-type analysis...")
    per_fault = {}
    for ft_idx in range(9):
        ft_mask = fault_types == ft_idx
        if ft_mask.sum() == 0:
            continue

        ft_indices = set(np.where(ft_mask)[0])
        ft_code_correct = len(code_correct & ft_indices)
        ft_log_correct = len(log_correct & ft_indices)
        ft_full_correct = len(full_correct & ft_indices)
        ft_total = len(ft_indices)

        ft_code_only = len((code_correct - log_correct) & ft_indices)
        ft_log_only = len((log_correct - code_correct) & ft_indices)

        name = fault_type_names[ft_idx]
        per_fault[name] = {
            'total': int(ft_total),
            'code_acc': round(ft_code_correct / ft_total, 3) if ft_total > 0 else 0,
            'log_acc': round(ft_log_correct / ft_total, 3) if ft_total > 0 else 0,
            'full_acc': round(ft_full_correct / ft_total, 3) if ft_total > 0 else 0,
            'code_unique': int(ft_code_only),
            'log_unique': int(ft_log_only),
            'complementarity': round((ft_code_only + ft_log_only) / ft_total, 3) if ft_total > 0 else 0,
        }
        print(f"    {name:15s}: code={per_fault[name]['code_acc']:.1%} "
              f"log={per_fault[name]['log_acc']:.1%} "
              f"full={per_fault[name]['full_acc']:.1%} "
              f"compl={per_fault[name]['complementarity']:.1%}")

    # 6. Save results
    results = {
        'venn_diagram': venn_data,
        'per_fault_type': per_fault,
        'model_accuracies': {
            'code_only': round(len(code_correct) / n_total, 4),
            'log_only': round(len(log_correct) / n_total, 4),
            'full_model': round(len(full_correct) / n_total, 4),
        },
    }

    output_path = RESULTS_DIR / "motivation_study.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved to {output_path}")

    return results


if __name__ == "__main__":
    main()
