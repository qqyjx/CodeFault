#!/usr/bin/env python3
"""
BugsInPy Data Processor

Processes bugs from the BugsInPy benchmark (493 bugs from 17 Python projects).

Data Source: https://github.com/soarsmu/BugsInPy

Output Format:
{
    "id": "bugsinpy_<project>_<bug_id>",
    "code_change": {
        "project": "<project_name>",
        "commit_sha": "<fix_commit>",
        "diff": "<unified_diff>",
        "file_path": "<modified_file>"
    },
    "log_context": {
        "messages": [],
        "source": "synthetic"
    },
    "label": 1,
    "source": "bugsinpy"
}
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


# BugsInPy projects and their bug counts
BUGSINPY_PROJECTS = {
    'ansible': 18,
    'black': 23,
    'cookiecutter': 4,
    'fastapi': 16,
    'httpie': 5,
    'keras': 45,
    'luigi': 33,
    'matplotlib': 30,
    'pandas': 169,
    'PySnooper': 3,
    'sanic': 5,
    'scrapy': 40,
    'spacy': 10,
    'thefuck': 32,
    'tornado': 16,
    'tqdm': 9,
    'youtube-dl': 35
}


def clone_bugsinpy(target_dir: Path) -> bool:
    """Clone BugsInPy repository."""
    if (target_dir / '.git').exists():
        print(f"BugsInPy already cloned at {target_dir}")
        return True

    print("Cloning BugsInPy repository...")
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1',
             'https://github.com/soarsmu/BugsInPy.git', str(target_dir)],
            check=True,
            capture_output=True,
            timeout=300
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Failed to clone BugsInPy: {e}")
        return False


def get_bug_info(bugsinpy_dir: Path, project: str, bug_id: int) -> Optional[Dict]:
    """Get bug information from BugsInPy."""
    bug_dir = bugsinpy_dir / 'projects' / project / 'bugs' / str(bug_id)

    if not bug_dir.exists():
        return None

    info = {
        'project': project,
        'bug_id': bug_id,
        'bug_dir': str(bug_dir)
    }

    # Read bug info file
    bug_info_file = bug_dir / 'bug.info'
    if bug_info_file.exists():
        with open(bug_info_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    info[key.strip()] = value.strip().strip('"')

    return info


def extract_diff_from_patch(bugsinpy_dir: Path, project: str, bug_id: int) -> Optional[str]:
    """Extract diff from bug patch file."""
    bug_dir = bugsinpy_dir / 'projects' / project / 'bugs' / str(bug_id)
    patch_file = bug_dir / 'bug_patch.txt'

    if not patch_file.exists():
        # Try alternative patch locations
        for alt_name in ['patch.diff', 'fix.patch', 'bug.patch']:
            alt_file = bug_dir / alt_name
            if alt_file.exists():
                patch_file = alt_file
                break

    if not patch_file.exists():
        return None

    try:
        with open(patch_file, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return None


def process_bug(bugsinpy_dir: Path, project: str, bug_id: int) -> Optional[Dict]:
    """Process a single BugsInPy bug."""
    info = get_bug_info(bugsinpy_dir, project, bug_id)
    if not info:
        return None

    diff = extract_diff_from_patch(bugsinpy_dir, project, bug_id)
    if not diff:
        return None

    # Extract modified files from diff
    modified_files = []
    for line in diff.split('\n'):
        if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
            parts = line.split()
            for part in parts:
                if part.endswith('.py'):
                    modified_files.append(part.lstrip('ab/'))

    return {
        "id": f"bugsinpy_{project}_{bug_id}",
        "code_change": {
            "project": project,
            "commit_sha": info.get('buggy_commit_id', ''),
            "fix_commit": info.get('fixed_commit_id', ''),
            "diff": diff,
            "files_changed": list(set(modified_files)),
            "additions": len([l for l in diff.split('\n') if l.startswith('+') and not l.startswith('+++')]),
            "deletions": len([l for l in diff.split('\n') if l.startswith('-') and not l.startswith('---')])
        },
        "log_context": {
            "messages": [],
            "timestamps": [],
            "source": "synthetic"
        },
        "label": 1,
        "source": "bugsinpy"
    }


def process_bugsinpy(
    output_dir: Path,
    bugsinpy_dir: Optional[Path] = None,
    projects: Optional[List[str]] = None,
    max_bugs_per_project: Optional[int] = None
) -> Dict:
    """
    Process BugsInPy bugs.

    Args:
        output_dir: Output directory
        bugsinpy_dir: Path to BugsInPy repo (will clone if not exists)
        projects: List of projects to process (None for all)
        max_bugs_per_project: Maximum bugs per project

    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("BugsInPy Data Processor")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone or use existing BugsInPy
    if bugsinpy_dir is None:
        bugsinpy_dir = output_dir.parent.parent / 'raw' / 'bugsinpy'

    bugsinpy_dir.mkdir(parents=True, exist_ok=True)

    if not clone_bugsinpy(bugsinpy_dir):
        print("Failed to get BugsInPy, generating placeholder data...")
        return generate_placeholder_data(output_dir, projects, max_bugs_per_project)

    # Select projects
    if projects is None:
        projects = list(BUGSINPY_PROJECTS.keys())

    print(f"Processing {len(projects)} projects")

    processed_samples = []
    project_counts = {}

    for project in projects:
        max_bugs = BUGSINPY_PROJECTS.get(project, 0)
        if max_bugs_per_project:
            max_bugs = min(max_bugs, max_bugs_per_project)

        print(f"\nProcessing {project} ({max_bugs} bugs)...")
        project_samples = []

        for bug_id in tqdm(range(1, max_bugs + 1), desc=project):
            sample = process_bug(bugsinpy_dir, project, bug_id)
            if sample:
                project_samples.append(sample)

        processed_samples.extend(project_samples)
        project_counts[project] = len(project_samples)
        print(f"  Processed {len(project_samples)} bugs from {project}")

    # Save results
    output_file = output_dir / "bugsinpy.json"
    with open(output_file, 'w') as f:
        json.dump(processed_samples, f, indent=2)

    stats = {
        'total_samples': len(processed_samples),
        'project_counts': project_counts
    }

    stats_file = output_dir / "bugsinpy_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved {len(processed_samples)} samples to {output_file}")
    return stats


def generate_placeholder_data(
    output_dir: Path,
    projects: Optional[List[str]] = None,
    max_bugs_per_project: Optional[int] = None
) -> Dict:
    """Generate placeholder data when BugsInPy is not available."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if projects is None:
        projects = list(BUGSINPY_PROJECTS.keys())

    samples = []
    for project in projects:
        max_bugs = BUGSINPY_PROJECTS.get(project, 0)
        if max_bugs_per_project:
            max_bugs = min(max_bugs, max_bugs_per_project)

        for bug_id in range(1, max_bugs + 1):
            samples.append({
                "id": f"bugsinpy_{project}_{bug_id}",
                "code_change": {
                    "project": project,
                    "commit_sha": f"placeholder_{project}_{bug_id}",
                    "diff": f"--- a/{project}/bug{bug_id}.py\n+++ b/{project}/bug{bug_id}.py\n@@ -10,1 +10,1 @@\n-    buggy_code()\n+    fixed_code()",
                    "files_changed": [f"{project}/bug{bug_id}.py"],
                    "additions": 1,
                    "deletions": 1
                },
                "log_context": {
                    "messages": [],
                    "source": "placeholder"
                },
                "label": 1,
                "source": "bugsinpy"
            })

    output_file = output_dir / "bugsinpy.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} placeholder samples")
    return {'total_samples': len(samples), 'is_placeholder': True}


def main():
    parser = argparse.ArgumentParser(description="Process BugsInPy dataset")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent.parent / "data" / "processed" / "codefault",
                       help="Output directory")
    parser.add_argument("--bugsinpy-dir", type=Path, default=None,
                       help="Path to BugsInPy repo")
    parser.add_argument("--projects", nargs='+', default=None,
                       help="Projects to process (default: all)")
    parser.add_argument("--max-bugs", type=int, default=None,
                       help="Maximum bugs per project")
    parser.add_argument("--placeholder", action="store_true",
                       help="Generate placeholder data")

    args = parser.parse_args()

    if args.placeholder:
        stats = generate_placeholder_data(
            args.output_dir,
            args.projects,
            args.max_bugs
        )
    else:
        stats = process_bugsinpy(
            output_dir=args.output_dir,
            bugsinpy_dir=args.bugsinpy_dir,
            projects=args.projects,
            max_bugs_per_project=args.max_bugs
        )

    print(f"\nDone! Processed {stats['total_samples']} samples")


if __name__ == "__main__":
    main()
