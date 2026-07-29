#!/usr/bin/env python3
"""
Normal Commits Miner for CodeFault

Mines normal (non-bug-fix) commits from GitHub repositories to serve as
negative samples for the fault prediction model.

Strategy:
1. Clone popular Java/Python repositories
2. Extract commits that are NOT bug-fixes (filter by commit message)
3. Generate unified diffs
4. Save in the same format as bug datasets

Output Format:
{
    "id": "normal_<project>_<sha>",
    "code_change": {
        "project": "<project_name>",
        "commit_sha": "<sha>",
        "diff": "<unified_diff>",
        "file_path": "<changed_file>",
        "additions": <int>,
        "deletions": <int>
    },
    "log_context": {
        "messages": [],
        "timestamps": [],
        "source": "none"
    },
    "label": 0,  # Normal commit = no bug
    "source": "github_normal"
}
"""

import os
import sys
import json
import argparse
import subprocess
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil


# Bug-related keywords to filter out
BUG_KEYWORDS = [
    'fix', 'bug', 'error', 'issue', 'crash', 'fault', 'defect', 'patch',
    'repair', 'resolve', 'correct', 'wrong', 'fail', 'broken', 'mistake',
    'typo', 'hotfix', 'regression', 'revert', 'rollback', 'CVE', 'vulnerability',
    'security', 'leak', 'overflow', 'null', 'NPE', 'exception', 'throws'
]

# Merge/CI keywords to filter out
SKIP_KEYWORDS = [
    'merge', 'Merge', 'MERGE', 'bump', 'version', 'release', 'changelog',
    'CI', 'travis', 'jenkins', 'github action', 'workflow', 'dependency',
    'upgrade', 'update dependency', 'renovate', 'dependabot'
]

# Popular GitHub repositories for mining
JAVA_REPOS = [
    "spring-projects/spring-boot",
    "elastic/elasticsearch",
    "apache/kafka",
    "google/guava",
    "ReactiveX/RxJava",
    "square/okhttp",
    "square/retrofit",
    "apache/dubbo",
    "netty/netty",
    "alibaba/fastjson",
]

PYTHON_REPOS = [
    "django/django",
    "pallets/flask",
    "psf/requests",
    "numpy/numpy",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "pytorch/pytorch",
    "huggingface/transformers",
    "ansible/ansible",
    "scrapy/scrapy",
]


def is_bug_fix_commit(message: str) -> bool:
    """Check if commit message indicates a bug fix."""
    message_lower = message.lower()
    for keyword in BUG_KEYWORDS:
        if keyword.lower() in message_lower:
            return True
    return False


def should_skip_commit(message: str) -> bool:
    """Check if commit should be skipped (merge, CI, etc.)."""
    for keyword in SKIP_KEYWORDS:
        if keyword.lower() in message.lower():
            return True
    return False


def is_valid_commit(message: str) -> bool:
    """Check if commit is a valid normal commit."""
    if not message or len(message.strip()) < 5:
        return False
    if is_bug_fix_commit(message):
        return False
    if should_skip_commit(message):
        return False
    return True


def clone_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone a repository (shallow clone for speed)."""
    try:
        # Shallow clone with limited history
        cmd = [
            'git', 'clone', '--depth', '1000', '--single-branch',
            f'https://github.com/{repo_url}.git', str(target_dir)
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
        return target_dir.exists()
    except Exception as e:
        print(f"Error cloning {repo_url}: {e}")
        return False


def get_commits_from_repo(repo_dir: Path, max_commits: int = 500) -> List[Dict]:
    """Get commit information from a local repository."""
    commits = []
    try:
        # Get commit log with format: sha|message|author|date
        cmd = [
            'git', '-C', str(repo_dir), 'log',
            f'--max-count={max_commits * 3}',  # Get more to filter
            '--format=%H|%s|%an|%ai',
            '--no-merges'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        for line in result.stdout.strip().split('\n'):
            if not line or '|' not in line:
                continue
            parts = line.split('|', 3)
            if len(parts) < 4:
                continue

            sha, message, author, date = parts
            if is_valid_commit(message):
                commits.append({
                    'sha': sha,
                    'message': message,
                    'author': author,
                    'date': date
                })

            if len(commits) >= max_commits:
                break

    except Exception as e:
        print(f"Error getting commits: {e}")

    return commits


def get_commit_diff(repo_dir: Path, sha: str) -> Optional[str]:
    """Get the unified diff for a commit."""
    try:
        cmd = [
            'git', '-C', str(repo_dir), 'show',
            '--format=', '--unified=3',
            '--diff-filter=M',  # Only modified files
            sha
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        diff = result.stdout.strip()

        # Validate diff
        if not diff or len(diff) < 10:
            return None
        if len(diff) > 50000:  # Skip very large diffs
            return None

        return diff
    except Exception as e:
        return None


def get_changed_files(repo_dir: Path, sha: str) -> List[str]:
    """Get list of changed files in a commit."""
    try:
        cmd = [
            'git', '-C', str(repo_dir), 'show',
            '--format=', '--name-only',
            sha
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return files
    except:
        return []


def count_diff_stats(diff: str) -> tuple:
    """Count additions and deletions in a diff."""
    additions = len([l for l in diff.split('\n') if l.startswith('+') and not l.startswith('+++')])
    deletions = len([l for l in diff.split('\n') if l.startswith('-') and not l.startswith('---')])
    return additions, deletions


def process_commit(repo_dir: Path, project_name: str, commit: Dict) -> Optional[Dict]:
    """Process a single commit into our format."""
    sha = commit['sha']

    # Get diff
    diff = get_commit_diff(repo_dir, sha)
    if not diff:
        return None

    # Get changed files
    files = get_changed_files(repo_dir, sha)
    if not files:
        return None

    # Filter by file extension (Java/Python only)
    code_files = [f for f in files if f.endswith(('.java', '.py'))]
    if not code_files:
        return None

    # Count stats
    additions, deletions = count_diff_stats(diff)

    # Skip trivial changes
    if additions + deletions < 3:
        return None

    return {
        "id": f"normal_{project_name}_{sha[:8]}",
        "code_change": {
            "project": project_name,
            "commit_sha": sha,
            "diff": diff,
            "file_path": code_files[0],
            "additions": additions,
            "deletions": deletions,
            "message": commit['message']
        },
        "log_context": {
            "messages": [],
            "timestamps": [],
            "source": "none"
        },
        "label": 0,  # Normal commit = no bug
        "source": "github_normal"
    }


def mine_repo(repo_url: str, output_dir: Path, max_samples: int = 500) -> List[Dict]:
    """Mine normal commits from a single repository."""
    project_name = repo_url.split('/')[-1]
    samples = []

    # Create temp directory for cloning
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = Path(tmp_dir) / project_name

        print(f"  Cloning {repo_url}...")
        if not clone_repo(repo_url, repo_dir):
            print(f"  Failed to clone {repo_url}")
            return []

        print(f"  Getting commits from {project_name}...")
        commits = get_commits_from_repo(repo_dir, max_commits=max_samples * 2)
        print(f"  Found {len(commits)} valid commits")

        # Process commits
        for commit in tqdm(commits[:max_samples], desc=f"  Processing {project_name}"):
            sample = process_commit(repo_dir, project_name, commit)
            if sample:
                samples.append(sample)

    return samples


def mine_normal_commits(
    output_dir: Path,
    repos: List[str] = None,
    max_samples_per_repo: int = 500,
    total_target: int = 30000,
    language: str = 'both'
) -> Dict:
    """
    Mine normal commits from multiple repositories.

    Args:
        output_dir: Directory to save output
        repos: List of repo URLs (owner/name format)
        max_samples_per_repo: Max samples per repository
        total_target: Total target number of samples
        language: 'java', 'python', or 'both'

    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select repositories
    if repos is None:
        if language == 'java':
            repos = JAVA_REPOS
        elif language == 'python':
            repos = PYTHON_REPOS
        else:
            repos = JAVA_REPOS + PYTHON_REPOS

    all_samples = []
    stats = {
        'repos_processed': 0,
        'repos_failed': 0,
        'total_samples': 0,
        'samples_per_repo': {}
    }

    print(f"\nMining normal commits from {len(repos)} repositories...")
    print(f"Target: {total_target} samples\n")

    for repo_url in repos:
        project_name = repo_url.split('/')[-1]
        print(f"\n[{stats['repos_processed'] + 1}/{len(repos)}] Processing {repo_url}")

        try:
            samples = mine_repo(repo_url, output_dir, max_samples_per_repo)
            if samples:
                all_samples.extend(samples)
                stats['samples_per_repo'][project_name] = len(samples)
                stats['repos_processed'] += 1
                print(f"  ✓ Got {len(samples)} samples from {project_name}")
            else:
                stats['repos_failed'] += 1
                print(f"  ✗ No samples from {project_name}")
        except Exception as e:
            stats['repos_failed'] += 1
            print(f"  ✗ Error processing {repo_url}: {e}")

        # Check if we have enough
        if len(all_samples) >= total_target:
            print(f"\nReached target of {total_target} samples")
            break

    # Shuffle and trim
    random.shuffle(all_samples)
    if len(all_samples) > total_target:
        all_samples = all_samples[:total_target]

    stats['total_samples'] = len(all_samples)

    # Save samples
    output_file = output_dir / "normal_commits.json"
    print(f"\nSaving {len(all_samples)} samples to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)

    # Save statistics
    stats_file = output_dir / "normal_commits_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"MINING COMPLETE")
    print(f"{'='*60}")
    print(f"Repositories processed: {stats['repos_processed']}")
    print(f"Repositories failed: {stats['repos_failed']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nTop contributors:")
    sorted_repos = sorted(stats['samples_per_repo'].items(), key=lambda x: -x[1])[:5]
    for repo, count in sorted_repos:
        print(f"  {repo}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Mine normal commits from GitHub")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("../data/processed/codefault"),
                       help="Output directory")
    parser.add_argument("--max-per-repo", type=int, default=500,
                       help="Max samples per repository")
    parser.add_argument("--target", type=int, default=30000,
                       help="Total target number of samples")
    parser.add_argument("--language", choices=['java', 'python', 'both'],
                       default='both', help="Language to mine")
    parser.add_argument("--repos", nargs='+', default=None,
                       help="Specific repos to mine (owner/name format)")

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir

    print("="*60)
    print("Normal Commits Miner for CodeFault")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    stats = mine_normal_commits(
        output_dir=output_dir,
        repos=args.repos,
        max_samples_per_repo=args.max_per_repo,
        total_target=args.target,
        language=args.language
    )

    print(f"\nDone! Mined {stats['total_samples']} normal commits")


if __name__ == "__main__":
    main()