#!/usr/bin/env python3
"""
Efficiency Analysis for CodeFault Experiments (RQ4)

Measures and compares:
- Model parameter count
- Inference latency (ms per sample)
- Memory footprint (MB)
- Training time (per epoch)
- Throughput (samples/sec)

Output: efficiency_results.json and efficiency_comparison.pdf
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_all_experiments import (
    CodeFault, DeepJIT, CC2Vec, JITFine, LAPredict, LogRobust,
    CodeFaultDataset, DEVICE, set_seed
)
from torch.utils.data import DataLoader


class EfficiencyProfiler:
    """Profile model efficiency metrics."""

    def __init__(self, device: torch.device = DEVICE):
        self.device = device

    def count_parameters(self, model: nn.Module) -> Dict:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'total_params_millions': total_params / 1e6,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # float32
        }

    def measure_inference_latency(
        self,
        model: nn.Module,
        sample_batch: Dict,
        n_warmup: int = 10,
        n_measure: int = 100
    ) -> Dict:
        """Measure inference latency."""
        model.eval()

        code_ids = sample_batch['code_ids'].to(self.device)
        log_ids = sample_batch['log_ids'].to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(code_ids, log_ids)

        # Synchronize if CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(n_measure):
                start = time.perf_counter()
                _ = model(code_ids, log_ids)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)  # ms

        latencies = np.array(latencies)
        batch_size = code_ids.size(0)

        return {
            'batch_size': batch_size,
            'n_measurements': n_measure,
            'latency_per_batch_ms': {
                'mean': float(latencies.mean()),
                'std': float(latencies.std()),
                'min': float(latencies.min()),
                'max': float(latencies.max()),
                'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99))
            },
            'latency_per_sample_ms': {
                'mean': float(latencies.mean() / batch_size),
                'std': float(latencies.std() / batch_size)
            },
            'throughput_samples_per_sec': float(batch_size * 1000 / latencies.mean())
        }

    def measure_memory_footprint(
        self,
        model: nn.Module,
        sample_batch: Dict
    ) -> Dict:
        """Measure GPU memory footprint during inference."""
        if self.device.type != 'cuda':
            return {
                'peak_memory_mb': 0,
                'allocated_memory_mb': 0,
                'note': 'Memory measurement requires CUDA'
            }

        model.eval()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        code_ids = sample_batch['code_ids'].to(self.device)
        log_ids = sample_batch['log_ids'].to(self.device)

        # Measure during inference
        with torch.no_grad():
            _ = model(code_ids, log_ids)

        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)

        return {
            'peak_memory_mb': float(peak_memory),
            'allocated_memory_mb': float(allocated_memory)
        }

    def measure_training_time(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        n_epochs: int = 3
    ) -> Dict:
        """Measure training time per epoch."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        epoch_times = []

        for epoch in range(n_epochs):
            start = time.time()

            for batch in train_loader:
                code_ids = batch['code_ids'].to(self.device)
                log_ids = batch['log_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                preds, _ = model(code_ids, log_ids)
                loss = criterion(preds.squeeze(), labels)
                loss.backward()
                optimizer.step()

            epoch_time = time.time() - start
            epoch_times.append(epoch_time)

        epoch_times = np.array(epoch_times)

        return {
            'n_epochs_measured': n_epochs,
            'samples_per_epoch': len(train_loader.dataset),
            'time_per_epoch_sec': {
                'mean': float(epoch_times.mean()),
                'std': float(epoch_times.std())
            },
            'time_per_sample_ms': float(epoch_times.mean() * 1000 / len(train_loader.dataset))
        }


def run_efficiency_analysis(
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    n_inference_runs: int = 100,
    n_training_epochs: int = 3
) -> Dict:
    """
    Run complete efficiency analysis for all models.

    Args:
        data_dir: Directory containing processed data
        output_dir: Directory for output files
        batch_size: Batch size for measurements
        n_inference_runs: Number of inference measurements
        n_training_epochs: Number of epochs for training time measurement

    Returns:
        Efficiency results dictionary
    """
    print("=" * 60)
    print("CodeFault Efficiency Analysis (RQ4)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_file = data_dir / "train.json"
    if not train_file.exists():
        print(f"Warning: {train_file} not found, using sample data")
        # Create minimal sample data for testing
        sample_data = [
            {
                "code_change": {"diff": "def foo(): pass"},
                "log_context": {"messages": ["INFO: test"]},
                "label": 1
            }
        ] * 100
        train_file.parent.mkdir(parents=True, exist_ok=True)
        with open(train_file, 'w') as f:
            json.dump(sample_data, f)

    train_dataset = CodeFaultDataset(train_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get sample batch
    sample_batch = next(iter(train_loader))
    vocab_size = len(train_dataset.vocab)

    print(f"\nDataset: {len(train_dataset)} samples")
    print(f"Vocab size: {vocab_size}")
    print(f"Batch size: {batch_size}")

    # Models to profile
    models_config = {
        'CodeFault': CodeFault,
        'DeepJIT': DeepJIT,
        'CC2Vec': CC2Vec,
        'JIT-Fine': JITFine,
        'LAPredict': LAPredict,
        'LogRobust': LogRobust
    }

    profiler = EfficiencyProfiler(device=DEVICE)
    results = {}

    for model_name, ModelClass in models_config.items():
        print(f"\n{'='*40}")
        print(f"Profiling: {model_name}")
        print('='*40)

        set_seed(42)
        model = ModelClass(vocab_size).to(DEVICE)

        # Parameter count
        print("  Counting parameters...")
        param_stats = profiler.count_parameters(model)
        print(f"    Total: {param_stats['total_params']:,} ({param_stats['total_params_millions']:.2f}M)")
        print(f"    Size: {param_stats['model_size_mb']:.2f} MB")

        # Inference latency
        print(f"  Measuring inference latency ({n_inference_runs} runs)...")
        latency_stats = profiler.measure_inference_latency(
            model, sample_batch, n_measure=n_inference_runs
        )
        print(f"    Per sample: {latency_stats['latency_per_sample_ms']['mean']:.3f} ± {latency_stats['latency_per_sample_ms']['std']:.3f} ms")
        print(f"    Throughput: {latency_stats['throughput_samples_per_sec']:.1f} samples/sec")

        # Memory footprint
        print("  Measuring memory footprint...")
        memory_stats = profiler.measure_memory_footprint(model, sample_batch)
        if memory_stats.get('peak_memory_mb', 0) > 0:
            print(f"    Peak memory: {memory_stats['peak_memory_mb']:.2f} MB")

        # Training time
        print(f"  Measuring training time ({n_training_epochs} epochs)...")
        # Re-initialize model for clean training measurement
        model = ModelClass(vocab_size).to(DEVICE)
        training_stats = profiler.measure_training_time(
            model, train_loader, n_epochs=n_training_epochs
        )
        print(f"    Per epoch: {training_stats['time_per_epoch_sec']['mean']:.2f} ± {training_stats['time_per_epoch_sec']['std']:.2f} sec")

        results[model_name] = {
            'parameters': param_stats,
            'inference': latency_stats,
            'memory': memory_stats,
            'training': training_stats
        }

    # Save results
    results_file = output_dir / "efficiency_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate summary table
    print("\n" + "=" * 80)
    print("EFFICIENCY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Params (M)':<12} {'Size (MB)':<12} {'Latency (ms)':<15} {'Throughput':<12}")
    print("-" * 80)

    for model_name, stats in results.items():
        params_m = stats['parameters']['total_params_millions']
        size_mb = stats['parameters']['model_size_mb']
        latency = stats['inference']['latency_per_sample_ms']['mean']
        throughput = stats['inference']['throughput_samples_per_sec']
        print(f"{model_name:<15} {params_m:<12.2f} {size_mb:<12.2f} {latency:<15.3f} {throughput:<12.1f}")

    # Generate LaTeX table
    latex_file = output_dir / "efficiency_table.tex"
    generate_latex_table(results, latex_file)
    print(f"\nLaTeX table saved to: {latex_file}")

    return results


def generate_latex_table(results: Dict, output_file: Path):
    """Generate LaTeX table for paper."""
    latex = r"""
\begin{table}[t]
\centering
\caption{RQ4: Efficiency comparison of fault prediction models}
\label{tab:efficiency}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Params (M)} & \textbf{Size (MB)} & \textbf{Latency (ms)} & \textbf{Throughput} \\
\midrule
"""
    for model_name, stats in results.items():
        params = stats['parameters']['total_params_millions']
        size = stats['parameters']['model_size_mb']
        latency = stats['inference']['latency_per_sample_ms']['mean']
        throughput = stats['inference']['throughput_samples_per_sec']

        if model_name == 'CodeFault':
            latex += f"\\textbf{{{model_name}}} & \\textbf{{{params:.2f}}} & \\textbf{{{size:.1f}}} & \\textbf{{{latency:.2f}}} & \\textbf{{{throughput:.0f}}} \\\\\n"
        else:
            latex += f"{model_name} & {params:.2f} & {size:.1f} & {latency:.2f} & {throughput:.0f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_file, 'w') as f:
        f.write(latex)


def main():
    parser = argparse.ArgumentParser(description="Run efficiency analysis")
    parser.add_argument("--data-dir", type=Path,
                       default=Path(__file__).parent.parent / "data" / "processed" / "codefault",
                       help="Directory containing processed data")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent.parent / "results",
                       help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-inference", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=3)

    args = parser.parse_args()

    results = run_efficiency_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_inference_runs=args.n_inference,
        n_training_epochs=args.n_epochs
    )


if __name__ == "__main__":
    main()
