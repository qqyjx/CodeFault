#!/usr/bin/env python3
"""
Dataset Merger for CodeFault

Merges data from multiple sources into final train/val/test splits:
- ManySStuBs4J (153K+ bug samples)
- Defects4J (854 bug samples)
- BugsInPy (493 bug samples)
- Bugs.jar (1,158 bug samples)
- Normal Commits (400K normal samples)

Output:
- train.json (70%)
- val.json (15%)
- test.json (15%)
- metadata.json (statistics)
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import hashlib


def load_json_data(file_path: Path) -> List[Dict]:
    """Load JSON data file."""
    if not file_path.exists():
        print(f"Warning: {file_path} not found, skipping")
        return []

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} samples from {file_path.name}")
    return data


def deduplicate_samples(samples: List[Dict]) -> List[Dict]:
    """Remove duplicate samples based on diff content."""
    seen = set()
    unique = []

    for sample in samples:
        # Create hash of diff content
        diff = sample.get('code_change', {}).get('diff', '')
        diff_hash = hashlib.md5(diff.encode()).hexdigest()

        if diff_hash not in seen:
            seen.add(diff_hash)
            unique.append(sample)

    removed = len(samples) - len(unique)
    if removed > 0:
        print(f"  Removed {removed} duplicate samples")

    return unique


def stratified_split(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = 'source',
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/val/test with stratification.

    Args:
        samples: List of samples
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        stratify_by: Key to stratify by ('source', 'label', 'project')
        seed: Random seed

    Returns:
        (train, val, test) lists
    """
    random.seed(seed)

    # Group by stratification key
    groups = defaultdict(list)
    for sample in samples:
        if stratify_by == 'source':
            key = sample.get('source', 'unknown')
        elif stratify_by == 'label':
            key = sample.get('label', 0)
        elif stratify_by == 'project':
            key = sample.get('code_change', {}).get('project', 'unknown')
        else:
            key = 'all'
        groups[key].append(sample)

    train, val, test = [], [], []

    for key, group_samples in groups.items():
        random.shuffle(group_samples)
        n = len(group_samples)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group_samples[:n_train])
        val.extend(group_samples[n_train:n_train + n_val])
        test.extend(group_samples[n_train + n_val:])

    # Final shuffle
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def compute_statistics(samples: List[Dict]) -> Dict:
    """Compute dataset statistics."""
    stats = {
        'total_samples': len(samples),
        'positive_samples': sum(1 for s in samples if s.get('label', 0) == 1),
        'negative_samples': sum(1 for s in samples if s.get('label', 0) == 0),
        'sources': defaultdict(int),
        'projects': defaultdict(int),
        'bug_types': defaultdict(int),
        'avg_diff_length': 0,
        'avg_log_messages': 0
    }

    total_diff_len = 0
    total_log_msgs = 0

    for sample in samples:
        # Source
        source = sample.get('source', 'unknown')
        stats['sources'][source] += 1

        # Project
        project = sample.get('code_change', {}).get('project', 'unknown')
        stats['projects'][project] += 1

        # Bug type
        bug_type = sample.get('code_change', {}).get('bug_type', 'unknown')
        if bug_type != 'unknown':
            stats['bug_types'][bug_type] += 1

        # Diff length
        diff = sample.get('code_change', {}).get('diff', '')
        total_diff_len += len(diff)

        # Log messages
        log_msgs = sample.get('log_context', {}).get('messages', [])
        total_log_msgs += len(log_msgs)

    if len(samples) > 0:
        stats['avg_diff_length'] = total_diff_len / len(samples)
        stats['avg_log_messages'] = total_log_msgs / len(samples)

    # Convert defaultdicts to regular dicts and limit top items
    stats['sources'] = dict(stats['sources'])
    stats['projects'] = dict(sorted(stats['projects'].items(), key=lambda x: -x[1])[:50])
    stats['bug_types'] = dict(sorted(stats['bug_types'].items(), key=lambda x: -x[1])[:20])

    stats['positive_ratio'] = stats['positive_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0

    return stats


def merge_datasets(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_samples_per_source: Optional[int] = None
) -> Dict:
    """
    Merge all data sources and create final splits.

    Args:
        data_dir: Directory containing processed data files
        output_dir: Output directory for final splits
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        max_samples_per_source: Maximum samples per source (for balancing)

    Returns:
        Metadata dictionary
    """
    print("=" * 60)
    print("CodeFault Dataset Merger")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Data source files
    source_files = {
        'manysstubs4j': data_dir / 'manysstubs4j.json',
        'defects4j': data_dir / 'defects4j.json',
        'bugsinpy': data_dir / 'bugsinpy.json',
        'bugsjar': data_dir / 'bugsjar.json',
        'normal_commits': data_dir / 'normal_commits.json',
    }

    # Load all data
    print("\nLoading data sources:")
    all_samples = []

    for source_name, file_path in source_files.items():
        samples = load_json_data(file_path)

        # Limit samples if specified
        if max_samples_per_source and len(samples) > max_samples_per_source:
            random.seed(seed)
            samples = random.sample(samples, max_samples_per_source)
            print(f"    Limited to {max_samples_per_source} samples")

        # Ensure source is set
        for s in samples:
            s['source'] = source_name

        all_samples.extend(samples)

    print(f"\nTotal samples loaded: {len(all_samples)}")

    # Deduplicate
    print("\nDeduplicating samples:")
    all_samples = deduplicate_samples(all_samples)
    print(f"Total after deduplication: {len(all_samples)}")

    # Split
    print(f"\nSplitting data (train={train_ratio}, val={val_ratio}, test={test_ratio}):")
    train, val, test = stratified_split(
        all_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by='source',
        seed=seed
    )

    print(f"  Train: {len(train)} samples")
    print(f"  Val: {len(val)} samples")
    print(f"  Test: {len(test)} samples")

    # Save splits
    print("\nSaving splits:")

    train_file = output_dir / 'train.json'
    with open(train_file, 'w') as f:
        json.dump(train, f)
    print(f"  Saved {train_file}")

    val_file = output_dir / 'val.json'
    with open(val_file, 'w') as f:
        json.dump(val, f)
    print(f"  Saved {val_file}")

    test_file = output_dir / 'test.json'
    with open(test_file, 'w') as f:
        json.dump(test, f)
    print(f"  Saved {test_file}")

    # Compute and save metadata
    print("\nComputing statistics:")
    metadata = {
        'created': datetime.now().isoformat(),
        'seed': seed,
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'train': compute_statistics(train),
        'val': compute_statistics(val),
        'test': compute_statistics(test),
        'total': compute_statistics(all_samples)
    }

    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {metadata_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"{'Split':<10} {'Total':<10} {'Positive':<10} {'Negative':<10} {'Pos Ratio':<10}")
    print("-" * 60)

    for split_name, split_stats in [('Train', metadata['train']),
                                     ('Val', metadata['val']),
                                     ('Test', metadata['test']),
                                     ('Total', metadata['total'])]:
        print(f"{split_name:<10} {split_stats['total_samples']:<10} "
              f"{split_stats['positive_samples']:<10} {split_stats['negative_samples']:<10} "
              f"{split_stats['positive_ratio']:.2%}")

    print("\nSource distribution (Total):")
    for source, count in sorted(metadata['total']['sources'].items(), key=lambda x: -x[1]):
        pct = count / metadata['total']['total_samples'] * 100
        print(f"  {source}: {count} ({pct:.1f}%)")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Merge datasets for CodeFault")
    parser.add_argument("--data-dir", type=Path,
                       default=Path(__file__).parent.parent / "data" / "processed" / "codefault",
                       help="Directory containing processed data files")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent.parent / "data" / "processed" / "codefault",
                       help="Output directory for merged data")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-source", type=int, default=None,
                       help="Maximum samples per source")

    args = parser.parse_args()

    metadata = merge_datasets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples_per_source=args.max_per_source
    )

    print(f"\nDone! Created dataset with {metadata['total']['total_samples']} samples")


if __name__ == "__main__":
    main()
