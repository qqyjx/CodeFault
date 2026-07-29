#!/usr/bin/env python3
"""
ManySStuBs4J Data Processor

Downloads and processes the ManySStuBs4J dataset (~153,652 single-statement bugs).

Data Source: https://zenodo.org/records/3653444
Paper: How Often Do Single-Statement Bugs Occur? (MSR 2019)

Output Format:
{
    "id": "manysstubs_<project>_<idx>",
    "code_change": {
        "project": "<project_name>",
        "commit_sha": "<fix_commit>",
        "diff": "<unified_diff>",
        "bug_type": "<sstub_type>",
        "file_path": "<bug_file_path>"
    },
    "log_context": {
        "messages": [],  # Will be populated later by link_loghub.py
        "source": "synthetic"
    },
    "label": 1,  # All are bug-inducing
    "source": "manysstubs4j"
}
"""

import os
import sys
import json
import argparse
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib


# ManySStuBs4J dataset URLs (Zenodo API format - files don't have .json extension)
MANYSSTUBS_URLS = {
    "sstubs": "https://zenodo.org/api/records/3653444/files/sstubs/content",
    "sstubsLarge": "https://zenodo.org/api/records/3653444/files/sstubsLarge/content",
}

# 16 SStuB bug types
SSTUB_TYPES = [
    "CHANGE_IDENTIFIER",
    "CHANGE_NUMERAL",
    "CHANGE_MODIFIER",
    "CHANGE_OPERATOR",
    "CHANGE_UNARY_OPERATOR",
    "CHANGE_OPERAND",
    "CHANGE_CALLER_IN_FUNCTION_CALL",
    "MORE_SPECIFIC_IF",
    "LESS_SPECIFIC_IF",
    "MISSING_THROWS_EXCEPTION",
    "DELETE_THROWS_EXCEPTION",
    "SWAP_ARGUMENTS",
    "SWAP_BOOLEAN_LITERAL",
    "OVERLOAD_METHOD_MORE_ARGS",
    "OVERLOAD_METHOD_DELETED_ARGS",
    "DIFFERENT_METHOD_SAME_ARGS"
]


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def generate_diff(source_before: str, source_after: str) -> str:
    """Generate unified diff from before/after source code."""
    before_lines = source_before.splitlines(keepends=True)
    after_lines = source_after.splitlines(keepends=True)

    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile='a/buggy.java',
        tofile='b/fixed.java',
        lineterm=''
    )
    return ''.join(diff)


def process_sstub_entry(entry: Dict, idx: int) -> Optional[Dict]:
    """Process a single SStuB entry into our format."""
    try:
        project_name = entry.get('projectName', 'unknown')
        fix_commit = entry.get('fixCommitSHA1', '')
        bug_file = entry.get('bugFilePath', '')
        bug_type = entry.get('bugType', 'unknown')

        source_before = entry.get('sourceBeforeFix', '')
        source_after = entry.get('sourceAfterFix', '')

        if not source_before or not source_after:
            return None

        # Generate diff
        diff = generate_diff(source_before, source_after)

        if not diff:
            return None

        return {
            "id": f"manysstubs_{project_name}_{idx}",
            "code_change": {
                "project": project_name,
                "commit_sha": fix_commit,
                "diff": diff,
                "bug_type": bug_type,
                "file_path": bug_file,
                "additions": len([l for l in diff.split('\n') if l.startswith('+') and not l.startswith('+++')]),
                "deletions": len([l for l in diff.split('\n') if l.startswith('-') and not l.startswith('---')])
            },
            "log_context": {
                "messages": [],
                "timestamps": [],
                "source": "synthetic"
            },
            "label": 1,
            "source": "manysstubs4j"
        }
    except Exception as e:
        return None


def process_manysstubs(
    data_dir: Path,
    output_dir: Path,
    use_large: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 4
) -> Dict:
    """
    Download and process ManySStuBs4J dataset.

    Args:
        data_dir: Directory to store raw data
        output_dir: Directory to store processed data
        use_large: Whether to use sstubsLarge.json (more samples)
        max_samples: Maximum number of samples to process (None for all)
        num_workers: Number of parallel workers

    Returns:
        Statistics dictionary
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select dataset
    dataset_key = "sstubsLarge" if use_large else "sstubs"
    url = MANYSSTUBS_URLS[dataset_key]
    json_path = data_dir / f"{dataset_key}.json"

    # Download if not exists
    if not json_path.exists():
        print(f"Downloading {dataset_key} dataset...")
        if not download_file(url, json_path, desc=f"Downloading {dataset_key}"):
            raise RuntimeError(f"Failed to download {dataset_key}")
    else:
        print(f"Using cached {json_path}")

    # Load data
    print("Loading JSON data...")
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw entries")

    # Limit samples if specified
    if max_samples:
        raw_data = raw_data[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Process entries
    print("Processing entries...")
    processed_samples = []
    bug_type_counts = {t: 0 for t in SSTUB_TYPES}
    project_counts = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_sstub_entry, entry, idx): idx
            for idx, entry in enumerate(raw_data)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result:
                processed_samples.append(result)

                # Count statistics
                bug_type = result['code_change'].get('bug_type', 'unknown')
                if bug_type in bug_type_counts:
                    bug_type_counts[bug_type] += 1

                project = result['code_change'].get('project', 'unknown')
                project_counts[project] = project_counts.get(project, 0) + 1

    print(f"Successfully processed {len(processed_samples)} samples")

    # Save processed data
    output_file = output_dir / "manysstubs4j.json"
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(processed_samples, f, indent=2)

    # Save statistics
    stats = {
        "total_samples": len(processed_samples),
        "source": dataset_key,
        "bug_type_distribution": bug_type_counts,
        "project_distribution": dict(sorted(project_counts.items(), key=lambda x: -x[1])[:20]),
        "unique_projects": len(project_counts)
    }

    stats_file = output_dir / "manysstubs4j_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Unique projects: {stats['unique_projects']}")
    print(f"  Top bug types:")
    for bug_type, count in sorted(bug_type_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    {bug_type}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Process ManySStuBs4J dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("../data/raw/manysstubs"),
                       help="Directory to store raw data")
    parser.add_argument("--output-dir", type=Path, default=Path("../data/processed/codefault"),
                       help="Directory to store processed data")
    parser.add_argument("--use-small", action="store_true",
                       help="Use smaller sstubs.json instead of sstubsLarge.json")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir

    stats = process_manysstubs(
        data_dir=data_dir,
        output_dir=output_dir,
        use_large=not args.use_small,
        max_samples=args.max_samples,
        num_workers=args.workers
    )

    print(f"\nDone! Processed {stats['total_samples']} samples")


if __name__ == "__main__":
    main()
