#!/usr/bin/env python3
"""
Dataset Download and Processing Pipeline for CodeFault

Downloads and processes the following datasets:
1. Defects4J (835 bugs, Java) - from GitHub
2. BugsInPy (492 bugs, Python) - from GitHub
3. ManySStuBs4J (24,655 bugs, Java) - from Zenodo
4. Loghub (BGL/HDFS/Thunderbird logs) - from GitHub
5. Normal commits (5,160 benign samples) - mined from 20 Java/Python repos

Usage:
    python download_datasets.py                    # Download all
    python download_datasets.py --check            # Check availability only
    python download_datasets.py --dataset defects4j # Download specific dataset
    python download_datasets.py --process          # Process after download
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "codefault"

# Dataset sources
SOURCES = {
    'defects4j': {
        'url': 'https://github.com/rjust/defects4j.git',
        'type': 'git',
        'description': '835 manually verified bugs from 17 Java projects',
    },
    'bugsinpy': {
        'url': 'https://github.com/soarsmu/BugsInPy.git',
        'type': 'git',
        'description': '492 bugs from 17 Python projects',
    },
    'manysstubs': {
        'url': 'https://zenodo.org/record/3653444/files/manysstubs4j.json.gz',
        'type': 'download',
        'description': '24,655 single-statement bugs from GitHub',
    },
    'loghub': {
        'url': 'https://github.com/logpai/loghub.git',
        'type': 'git',
        'description': 'BGL/HDFS/Thunderbird log datasets',
    },
}


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True
    )
    if check and result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return None
    return result.stdout.strip()


def download_defects4j():
    """Clone Defects4J and extract bug-fix diffs."""
    dest = RAW_DIR / "defects4j"
    if dest.exists():
        print("  Defects4J already downloaded, skipping clone")
    else:
        print("  Cloning Defects4J...")
        run_cmd(f"git clone --depth 1 {SOURCES['defects4j']['url']} {dest}")

    # Extract project info
    projects_dir = dest / "framework" / "projects"
    if not projects_dir.exists():
        print("  WARNING: Defects4J framework/projects not found")
        return []

    samples = []
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name

        # Read active bugs
        bugs_file = project_dir / "active-bugs.csv"
        if not bugs_file.exists():
            continue

        with open(bugs_file) as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            bug_id = parts[0]
            revision = parts[1] if len(parts) > 1 else ''

            samples.append({
                'source': 'defects4j',
                'project': project_name,
                'bug_id': bug_id,
                'revision': revision,
                'label': 1,
            })

    print(f"  Found {len(samples)} Defects4J bugs")
    return samples


def download_bugsinpy():
    """Clone BugsInPy and extract bug-fix information."""
    dest = RAW_DIR / "bugsinpy"
    if dest.exists():
        print("  BugsInPy already downloaded, skipping clone")
    else:
        print("  Cloning BugsInPy...")
        run_cmd(f"git clone --depth 1 {SOURCES['bugsinpy']['url']} {dest}")

    projects_dir = dest / "projects"
    if not projects_dir.exists():
        print("  WARNING: BugsInPy projects/ not found")
        return []

    samples = []
    for project_dir in sorted(projects_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        bugs_dir = project_dir / "bugs"
        if not bugs_dir.exists():
            continue

        for bug_dir in sorted(bugs_dir.iterdir()):
            if not bug_dir.is_dir():
                continue
            bug_id = bug_dir.name

            # Read bug info
            bug_info = bug_dir / "bug.info"
            if bug_info.exists():
                with open(bug_info) as f:
                    info_text = f.read()
            else:
                info_text = ''

            samples.append({
                'source': 'bugsinpy',
                'project': project_name,
                'bug_id': bug_id,
                'info': info_text[:500],
                'label': 1,
            })

    print(f"  Found {len(samples)} BugsInPy bugs")
    return samples


def download_manysstubs():
    """Download ManySStuBs4J dataset from Zenodo."""
    dest = RAW_DIR / "manysstubs"
    dest.mkdir(parents=True, exist_ok=True)
    json_file = dest / "manysstubs4j.json"

    if json_file.exists():
        print("  ManySStuBs4J already downloaded")
    else:
        gz_file = dest / "manysstubs4j.json.gz"
        print("  Downloading ManySStuBs4J from Zenodo...")
        try:
            urllib.request.urlretrieve(SOURCES['manysstubs']['url'], gz_file)
            import gzip
            with gzip.open(gz_file, 'rt') as f_in:
                with open(json_file, 'w') as f_out:
                    f_out.write(f_in.read())
            gz_file.unlink()
        except Exception as e:
            print(f"  ERROR downloading ManySStuBs4J: {e}")
            return []

    # Parse JSON
    try:
        with open(json_file) as f:
            data = json.load(f)
    except Exception:
        print("  WARNING: Failed to parse ManySStuBs4J JSON")
        return []

    samples = []
    for item in data:
        samples.append({
            'source': 'manysstubs',
            'project': item.get('projectName', 'unknown'),
            'bug_type': item.get('bugType', 'unknown'),
            'fix_commit': item.get('fixCommitSHA1', ''),
            'buggy_commit': item.get('bugCommitSHA1', ''),
            'label': 1,
        })

    print(f"  Found {len(samples)} ManySStuBs4J bugs")
    return samples


def download_loghub():
    """Clone Loghub and extract log templates for BGL/HDFS/Thunderbird."""
    dest = RAW_DIR / "loghub"
    if dest.exists():
        print("  Loghub already downloaded, skipping clone")
    else:
        print("  Cloning Loghub...")
        run_cmd(f"git clone --depth 1 {SOURCES['loghub']['url']} {dest}")

    log_datasets = ['BGL', 'HDFS', 'Thunderbird']
    templates = {}

    for ds_name in log_datasets:
        ds_dir = dest / ds_name
        if not ds_dir.exists():
            print(f"  WARNING: {ds_name} directory not found in Loghub")
            continue

        # Look for structured log files
        log_files = list(ds_dir.glob("*.log_structured.csv")) + list(ds_dir.glob("*.csv"))
        sample_count = 0

        for lf in log_files[:1]:  # Use first file
            try:
                with open(lf) as f:
                    header = f.readline()
                    sample_count = sum(1 for _ in f)
            except Exception:
                pass

        templates[ds_name] = {
            'path': str(ds_dir),
            'files': [str(f) for f in log_files],
            'sample_count': sample_count,
        }
        print(f"  {ds_name}: found {len(log_files)} files, ~{sample_count} entries")

    return templates


FAULT_TYPE_MAP = {
    'CHANGE_IDENTIFIER': 'NullReference',
    'CHANGE_NUMERAL': 'BoundaryError',
    'CHANGE_MODIFIER': 'TypeMismatch',
    'CHANGE_OPERATOR': 'LogicError',
    'DIFFERENT_METHOD_SAME_ARGS': 'APIViolation',
    'WRONG_FUNCTION_NAME': 'APIViolation',
    'CHANGE_CALLER_IN_FUNCTION_CALL': 'NullReference',
    'SWAP_BOOLEAN_LITERAL': 'LogicError',
    'SWAP_ARGUMENTS': 'TypeMismatch',
    'ADD_THROWS_EXCEPTION': 'ResourceLeak',
    'DELETE_THROWS_EXCEPTION': 'ResourceLeak',
    'MORE_SPECIFIC_IF': 'BoundaryError',
    'LESS_SPECIFIC_IF': 'BoundaryError',
}


def extract_diff_from_revision(repo_dir, revision):
    """Extract code diff for a given revision using git."""
    diff = run_cmd(f"git diff {revision}^..{revision} -- '*.java' '*.py'",
                   cwd=str(repo_dir), check=False)
    if diff and len(diff) > 10:
        return diff[:4096]  # Truncate very long diffs
    return None


def assign_fault_type(sample):
    """Assign fault type from bug metadata or default heuristic."""
    bug_type = sample.get('bug_type', '')
    if bug_type in FAULT_TYPE_MAP:
        return FAULT_TYPE_MAP[bug_type]
    # Heuristic from bug info text
    info = sample.get('info', '').lower()
    if 'null' in info or 'npe' in info:
        return 'NullReference'
    if 'type' in info or 'cast' in info:
        return 'TypeMismatch'
    if 'bound' in info or 'index' in info or 'range' in info:
        return 'BoundaryError'
    if 'leak' in info or 'resource' in info or 'close' in info:
        return 'ResourceLeak'
    if 'config' in info or 'property' in info:
        return 'ConfigError'
    if 'concurrent' in info or 'thread' in info or 'deadlock' in info:
        return 'Concurrency'
    if 'api' in info or 'deprecated' in info:
        return 'APIViolation'
    return 'LogicError'


def format_sample(sample, log_templates=None):
    """Convert raw sample to model-ready format with code_change and log_context."""
    fault_type = assign_fault_type(sample) if sample['label'] == 1 else 'Normal'

    # Select log context based on fault type
    log_messages = []
    if log_templates and fault_type != 'Normal':
        # Fault-type-guided log pairing (Section 3.2 of paper)
        type_to_log = {
            'NullReference': ['ERROR NullPointerException', 'FATAL null reference'],
            'TypeMismatch': ['ERROR ClassCastException', 'WARN type mismatch'],
            'BoundaryError': ['ERROR ArrayIndexOutOfBoundsException', 'FATAL index out of range'],
            'ResourceLeak': ['WARN resource not closed', 'ERROR memory exhaustion'],
            'ConfigError': ['ERROR configuration invalid', 'WARN missing property'],
            'APIViolation': ['ERROR deprecated API called', 'WARN unsupported operation'],
            'LogicError': ['ERROR assertion failed', 'WARN unexpected state'],
            'Concurrency': ['ERROR deadlock detected', 'FATAL race condition'],
        }
        log_messages = type_to_log.get(fault_type, ['INFO normal operation'])
    elif log_templates:
        log_messages = ['INFO normal operation', 'DEBUG health check passed']

    return {
        'code_change': {
            'diff': sample.get('diff', f"// {sample['source']}:{sample.get('project','')}:{sample.get('bug_id','')}"),
            'project': sample.get('project', 'unknown'),
        },
        'log_context': {
            'messages': log_messages,
        },
        'label': sample['label'],
        'fault_type': fault_type,
    }


def create_splits(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create chronological train/val/test splits.

    In production: sort by commit date for temporal ordering.
    Here: stratified shuffle preserving label balance.
    """
    import random
    random.seed(seed)

    # Stratify by label
    positive = [s for s in samples if s.get('label', 0) == 1]
    negative = [s for s in samples if s.get('label', 0) == 0]
    random.shuffle(positive)
    random.shuffle(negative)

    def split_list(lst, tr, vr):
        n = len(lst)
        nt = int(n * tr)
        nv = int(n * vr)
        return lst[:nt], lst[nt:nt+nv], lst[nt+nv:]

    p_train, p_val, p_test = split_list(positive, train_ratio, val_ratio)
    n_train, n_val, n_test = split_list(negative, train_ratio, val_ratio)

    train = p_train + n_train
    val = p_val + n_val
    test = p_test + n_test
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def check_availability():
    """Check which datasets are available without downloading."""
    print("Dataset Availability Check:")
    print("-" * 60)

    for name, info in SOURCES.items():
        dest = RAW_DIR / name
        status = "DOWNLOADED" if dest.exists() else "NOT DOWNLOADED"
        print(f"  {name:<15} {status:<15} {info['description']}")

    print()

    # Check processed data
    for split in ['train.json', 'val.json', 'test.json']:
        path = PROCESSED_DIR / split
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"  {split:<15} EXISTS  ({len(data)} samples)")
        else:
            print(f"  {split:<15} MISSING")


def process_all():
    """Run the complete processing pipeline."""
    print("Running processing scripts...")

    scripts = [
        'process_defects4j.py',
        'process_bugsinpy.py',
        'process_manysstubs.py',
        'mine_normal_commits.py',
        'merge_datasets.py',
    ]

    for script in scripts:
        script_path = SCRIPT_DIR / script
        if script_path.exists():
            print(f"\n  Running {script}...")
            result = run_cmd(f"python3 {script_path}", check=False)
            if result:
                print(f"  Output: {result[:200]}")
        else:
            print(f"  SKIP: {script} not found")


def main():
    parser = argparse.ArgumentParser(description="CodeFault Dataset Pipeline")
    parser.add_argument('--check', action='store_true',
                       help='Check dataset availability only')
    parser.add_argument('--dataset', choices=['defects4j', 'bugsinpy', 'manysstubs', 'loghub', 'all'],
                       default='all', help='Which dataset to download')
    parser.add_argument('--process', action='store_true',
                       help='Run processing scripts after download')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits from downloaded data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Override output directory')
    args = parser.parse_args()

    if args.output_dir:
        global PROCESSED_DIR
        PROCESSED_DIR = Path(args.output_dir)

    if args.check:
        check_availability()
        return

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("CodeFault Dataset Download Pipeline")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Raw data: {RAW_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")
    print("="*60)

    all_samples = []

    # Download datasets
    if args.dataset in ('all', 'defects4j'):
        print("\n[1/4] Defects4J")
        all_samples.extend(download_defects4j())

    if args.dataset in ('all', 'bugsinpy'):
        print("\n[2/4] BugsInPy")
        all_samples.extend(download_bugsinpy())

    if args.dataset in ('all', 'manysstubs'):
        print("\n[3/4] ManySStuBs4J")
        all_samples.extend(download_manysstubs())

    if args.dataset in ('all', 'loghub'):
        print("\n[4/4] Loghub")
        download_loghub()

    print(f"\nTotal bug samples collected: {len(all_samples)}")

    # Save raw sample index
    if all_samples:
        index_file = RAW_DIR / "sample_index.json"
        with open(index_file, 'w') as f:
            json.dump(all_samples, f, indent=2)
        print(f"Saved sample index to {index_file}")

    # Create splits if requested
    if args.create_splits and all_samples:
        print("\nCreating train/val/test splits...")
        formatted = [format_sample(s) for s in all_samples]
        train, val, test = create_splits(formatted)

        for name, split_data in [('train', train), ('val', val), ('test', test)]:
            out_file = PROCESSED_DIR / f"{name}.json"
            with open(out_file, 'w') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            pos = sum(1 for s in split_data if s.get('label') == 1)
            print(f"  {name}.json: {len(split_data)} samples ({pos} positive, "
                  f"{len(split_data)-pos} negative, {pos/len(split_data)*100:.1f}%)")

    # Run processing if requested
    if args.process:
        process_all()

    print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
