#!/usr/bin/env python3
"""
Defects4J Data Processor

Processes bugs from the Defects4J benchmark (854 bugs from 17 Java projects).

Prerequisites:
- Java 11+
- Git
- SVN
- Perl

Data Source: https://github.com/rjust/defects4j

Output Format:
{
    "id": "defects4j_<project>_<bug_id>",
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
    "source": "defects4j"
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
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# Defects4J projects and their bug counts
DEFECTS4J_PROJECTS = {
    'Chart': 26,
    'Cli': 39,
    'Closure': 174,
    'Codec': 18,
    'Collections': 4,
    'Compress': 47,
    'Csv': 16,
    'Gson': 18,
    'JacksonCore': 26,
    'JacksonDatabind': 112,
    'JacksonXml': 6,
    'Jsoup': 93,
    'JxPath': 22,
    'Lang': 64,
    'Math': 106,
    'Mockito': 38,
    'Time': 26
}


def check_defects4j_installed() -> bool:
    """Check if Defects4J is installed and configured."""
    try:
        result = subprocess.run(
            ['defects4j', 'info', '-p', 'Chart'],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_bug_info(project: str, bug_id: int) -> Optional[Dict]:
    """Get bug information using defects4j export."""
    try:
        # Get bug report
        result = subprocess.run(
            ['defects4j', 'info', '-p', project, '-b', str(bug_id)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return None

        info = {'project': project, 'bug_id': bug_id, 'raw_info': result.stdout}

        # Parse revision info
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Revision ID (fixed version)' in line:
                info['fix_revision'] = line.split(':')[-1].strip()
            elif 'Revision ID (buggy version)' in line:
                info['buggy_revision'] = line.split(':')[-1].strip()

        return info
    except subprocess.TimeoutExpired:
        return None


def extract_diff(project: str, bug_id: int, work_dir: Path) -> Optional[str]:
    """Extract unified diff for a bug by checking out buggy and fixed versions."""
    buggy_dir = work_dir / f"{project}_{bug_id}_buggy"
    fixed_dir = work_dir / f"{project}_{bug_id}_fixed"

    try:
        # Checkout buggy version
        subprocess.run(
            ['defects4j', 'checkout', '-p', project, '-v', f'{bug_id}b', '-w', str(buggy_dir)],
            capture_output=True,
            timeout=300
        )

        # Checkout fixed version
        subprocess.run(
            ['defects4j', 'checkout', '-p', project, '-v', f'{bug_id}f', '-w', str(fixed_dir)],
            capture_output=True,
            timeout=300
        )

        if not buggy_dir.exists() or not fixed_dir.exists():
            return None

        # Generate diff
        result = subprocess.run(
            ['diff', '-ruN', str(buggy_dir), str(fixed_dir)],
            capture_output=True,
            text=True,
            timeout=60
        )

        diff = result.stdout

        # Clean up
        shutil.rmtree(buggy_dir, ignore_errors=True)
        shutil.rmtree(fixed_dir, ignore_errors=True)

        return diff if diff else None

    except subprocess.TimeoutExpired:
        shutil.rmtree(buggy_dir, ignore_errors=True)
        shutil.rmtree(fixed_dir, ignore_errors=True)
        return None


def process_bug(project: str, bug_id: int, work_dir: Path) -> Optional[Dict]:
    """Process a single Defects4J bug."""
    info = get_bug_info(project, bug_id)
    if not info:
        return None

    diff = extract_diff(project, bug_id, work_dir)
    if not diff:
        return None

    # Extract modified files from diff
    modified_files = []
    for line in diff.split('\n'):
        if line.startswith('diff -ruN') or line.startswith('---') or line.startswith('+++'):
            parts = line.split()
            if len(parts) >= 2:
                file_path = parts[-1]
                if file_path.endswith('.java'):
                    modified_files.append(file_path)

    return {
        "id": f"defects4j_{project}_{bug_id}",
        "code_change": {
            "project": project,
            "commit_sha": info.get('fix_revision', ''),
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
        "source": "defects4j"
    }


def process_defects4j(
    output_dir: Path,
    projects: Optional[List[str]] = None,
    max_bugs_per_project: Optional[int] = None,
    num_workers: int = 1
) -> Dict:
    """
    Process Defects4J bugs.

    Args:
        output_dir: Output directory
        projects: List of projects to process (None for all)
        max_bugs_per_project: Maximum bugs per project
        num_workers: Number of parallel workers (1 recommended for Defects4J)

    Returns:
        Statistics dictionary
    """
    print("=" * 60)
    print("Defects4J Data Processor")
    print("=" * 60)

    # Check installation
    if not check_defects4j_installed():
        print("ERROR: Defects4J not found or not properly configured.")
        print("Please install Defects4J from: https://github.com/rjust/defects4j")
        print("\nFor now, generating placeholder data...")
        return generate_placeholder_data(output_dir, projects, max_bugs_per_project)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Select projects
    if projects is None:
        projects = list(DEFECTS4J_PROJECTS.keys())

    print(f"Processing {len(projects)} projects")

    # Create work directory
    work_dir = Path(tempfile.mkdtemp(prefix='defects4j_'))
    print(f"Work directory: {work_dir}")

    processed_samples = []
    project_counts = {}

    try:
        for project in projects:
            max_bugs = DEFECTS4J_PROJECTS.get(project, 0)
            if max_bugs_per_project:
                max_bugs = min(max_bugs, max_bugs_per_project)

            print(f"\nProcessing {project} ({max_bugs} bugs)...")
            project_samples = []

            for bug_id in tqdm(range(1, max_bugs + 1), desc=project):
                sample = process_bug(project, bug_id, work_dir)
                if sample:
                    project_samples.append(sample)

            processed_samples.extend(project_samples)
            project_counts[project] = len(project_samples)
            print(f"  Processed {len(project_samples)} bugs from {project}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Save results
    output_file = output_dir / "defects4j.json"
    with open(output_file, 'w') as f:
        json.dump(processed_samples, f, indent=2)

    stats = {
        'total_samples': len(processed_samples),
        'project_counts': project_counts
    }

    stats_file = output_dir / "defects4j_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved {len(processed_samples)} samples to {output_file}")
    return stats


def generate_placeholder_data(
    output_dir: Path,
    projects: Optional[List[str]] = None,
    max_bugs_per_project: Optional[int] = None
) -> Dict:
    """Generate placeholder data when Defects4J is not installed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if projects is None:
        projects = list(DEFECTS4J_PROJECTS.keys())

    samples = []
    for project in projects:
        max_bugs = DEFECTS4J_PROJECTS.get(project, 0)
        if max_bugs_per_project:
            max_bugs = min(max_bugs, max_bugs_per_project)

        for bug_id in range(1, max_bugs + 1):
            samples.append({
                "id": f"defects4j_{project}_{bug_id}",
                "code_change": {
                    "project": project,
                    "commit_sha": f"placeholder_{project}_{bug_id}",
                    "diff": f"--- a/src/{project}/Bug{bug_id}.java\n+++ b/src/{project}/Bug{bug_id}.java\n@@ -10,1 +10,1 @@\n-    buggy code\n+    fixed code",
                    "files_changed": [f"src/{project}/Bug{bug_id}.java"],
                    "additions": 1,
                    "deletions": 1
                },
                "log_context": {
                    "messages": [],
                    "source": "placeholder"
                },
                "label": 1,
                "source": "defects4j"
            })

    output_file = output_dir / "defects4j.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {len(samples)} placeholder samples")
    return {'total_samples': len(samples), 'is_placeholder': True}


def main():
    parser = argparse.ArgumentParser(description="Process Defects4J dataset")
    parser.add_argument("--output-dir", type=Path,
                       default=Path(__file__).parent.parent / "data" / "processed" / "codefault",
                       help="Output directory")
    parser.add_argument("--projects", nargs='+', default=None,
                       help="Projects to process (default: all)")
    parser.add_argument("--max-bugs", type=int, default=None,
                       help="Maximum bugs per project")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of workers (1 recommended)")
    parser.add_argument("--placeholder", action="store_true",
                       help="Generate placeholder data without Defects4J")

    args = parser.parse_args()

    if args.placeholder:
        stats = generate_placeholder_data(
            args.output_dir,
            args.projects,
            args.max_bugs
        )
    else:
        stats = process_defects4j(
            output_dir=args.output_dir,
            projects=args.projects,
            max_bugs_per_project=args.max_bugs,
            num_workers=args.workers
        )

    print(f"\nDone! Processed {stats['total_samples']} samples")


if __name__ == "__main__":
    main()
