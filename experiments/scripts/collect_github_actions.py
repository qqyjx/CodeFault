#!/usr/bin/env python3
"""
GitHub Actions Real CI/CD Data Collector

Collects real code-log pairs from public GitHub Actions workflow runs.
For each failed CI run, extracts:
  - The commit diff (code change)
  - The CI log output (runtime log)
  - Failure metadata (fault type label)

This produces a gold-standard evaluation set for construct validity
validation of CodeFault's synthetic code-log pairing strategy.

Usage:
    # Collect from curated repo list (recommended)
    python collect_github_actions.py --mode curated --output ../data/real_ci/

    # Search and collect from top Java/Python repos
    python collect_github_actions.py --mode search --language java --max-repos 50

    # Resume interrupted collection
    python collect_github_actions.py --mode curated --output ../data/real_ci/ --resume

Requirements:
    - GitHub personal access token (GITHUB_TOKEN env var)
    - requests, tqdm
"""

import os
import sys
import json
import re
import time
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import requests
from tqdm import tqdm


# --- Configuration ---

GITHUB_API = "https://api.github.com"
MAX_DIFF_SIZE = 50_000       # Skip huge diffs (>50KB)
MAX_LOG_SIZE = 200_000       # Truncate logs to 200KB
MIN_DIFF_LINES = 3           # Skip trivial 1-line changes
MAX_DIFF_FILES = 20          # Skip massive multi-file changes
RATE_LIMIT_PAUSE = 60        # Seconds to wait on rate limit
REQUEST_TIMEOUT = 30         # Seconds per request

# Curated list of well-maintained repos with active CI and known bug histories
CURATED_REPOS = [
    # Java projects (matching Defects4J ecosystem) — 25 repos
    "apache/commons-lang",
    "apache/commons-math",
    "apache/commons-io",
    "apache/commons-collections",
    "apache/commons-csv",
    "apache/commons-text",
    "google/gson",
    "google/guava",
    "square/okhttp",
    "square/retrofit",
    "ReactiveX/RxJava",
    "junit-team/junit5",
    "mockito/mockito",
    "spring-projects/spring-framework",
    "spring-projects/spring-boot",
    "elastic/elasticsearch",
    "apache/kafka",
    "apache/flink",
    "apache/maven",
    "apache/tomcat",
    "eclipse/jetty.project",
    "netty/netty",
    "mybatis/mybatis-3",
    "alibaba/fastjson2",
    "apache/dubbo",
    # Python projects (matching BugsInPy ecosystem) — 30 repos
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "psf/requests",
    "pallets/flask",
    "django/django",
    "pypa/pip",
    "python/cpython",
    "numpy/numpy",
    "matplotlib/matplotlib",
    "pytest-dev/pytest",
    "celery/celery",
    "tornadoweb/tornado",
    "scrapy/scrapy",
    "ansible/ansible",
    "saltstack/salt",
    "huggingface/transformers",
    "pytorch/pytorch",
    "python-poetry/poetry",
    "psf/black",
    "astral-sh/ruff",
    "pydantic/pydantic",
    "tiangolo/fastapi",
    "encode/httpx",
    "aio-libs/aiohttp",
    "sympy/sympy",
    "pallets/werkzeug",
    "pallets/jinja",
    "arrow-py/arrow",
    "boto/boto3",
    "python-pillow/Pillow",
]

# Fault type classification patterns for CI logs
FAULT_PATTERNS = {
    "NullReference": [
        r"NullPointerException",
        r"AttributeError: 'NoneType'",
        r"TypeError: Cannot read propert.*null",
        r"null reference",
    ],
    "TypeMismatch": [
        r"ClassCastException",
        r"TypeError:",
        r"incompatible types",
        r"type mismatch",
    ],
    "BoundaryError": [
        r"IndexOutOfBoundsException",
        r"ArrayIndexOutOfBoundsException",
        r"IndexError:",
        r"KeyError:",
        r"StringIndexOutOfBoundsException",
    ],
    "ResourceLeak": [
        r"OutOfMemoryError",
        r"MemoryError",
        r"Too many open files",
        r"Resource leak",
    ],
    "ConfigError": [
        r"FileNotFoundException",
        r"NoSuchFileException",
        r"ModuleNotFoundError",
        r"ImportError",
        r"ClassNotFoundException",
    ],
    "APIViolation": [
        r"IllegalArgumentException",
        r"IllegalStateException",
        r"ValueError:",
        r"AssertionError",
    ],
    "LogicError": [
        r"AssertionError",
        r"assert.*failed",
        r"Expected.*but got",
        r"FAIL:",
        r"FAILED",
    ],
    "Concurrency": [
        r"DeadlockException",
        r"TimeoutException",
        r"ConcurrentModificationException",
        r"deadlock",
        r"race condition",
    ],
}


@dataclass
class CICodeLogPair:
    """A real code-log pair from CI/CD."""
    id: str
    repo: str
    commit_sha: str
    commit_message: str
    code_change: Dict  # {diff, files_changed, additions, deletions}
    log_context: Dict  # {messages, source, workflow_name, job_name}
    fault_type: str    # Classified from log patterns
    label: int         # 1 = fault-inducing, 0 = benign
    metadata: Dict     # {run_id, run_url, created_at, language}


class GitHubActionsCollector:
    """Collects real CI/CD code-log pairs from GitHub Actions."""

    def __init__(self, token: str, output_dir: Path):
        self.token = token
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        self.collected: List[CICodeLogPair] = []
        self._load_checkpoint()

    def _api_get(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make a rate-limit-aware API request."""
        for attempt in range(3):
            try:
                resp = self.session.get(
                    url, params=params, timeout=REQUEST_TIMEOUT
                )

                # Handle rate limiting
                if resp.status_code == 403:
                    remaining = int(resp.headers.get("X-RateLimit-Remaining", 0))
                    if remaining == 0:
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                        wait = max(reset_time - time.time(), RATE_LIMIT_PAUSE)
                        print(f"\n  Rate limited. Waiting {wait:.0f}s...")
                        time.sleep(wait)
                        continue

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None
                else:
                    print(f"\n  API error {resp.status_code}: {resp.text[:200]}")
                    time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                print(f"\n  Request error: {e}")
                time.sleep(2 ** attempt)

        return None

    def _api_get_text(self, url: str) -> Optional[str]:
        """Get raw text content (for logs)."""
        for attempt in range(3):
            try:
                headers = {
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3.raw",
                }
                resp = requests.get(
                    url, headers=headers, timeout=REQUEST_TIMEOUT
                )
                if resp.status_code == 200:
                    return resp.text[:MAX_LOG_SIZE]
                elif resp.status_code == 403:
                    time.sleep(RATE_LIMIT_PAUSE)
                    continue
            except requests.exceptions.RequestException:
                time.sleep(2 ** attempt)
        return None

    def _classify_fault_type(self, log_text: str) -> str:
        """Classify fault type from CI log content."""
        log_lower = log_text.lower()

        # Count pattern matches per fault type
        scores = {}
        for fault_type, patterns in FAULT_PATTERNS.items():
            score = sum(
                len(re.findall(p, log_text, re.IGNORECASE))
                for p in patterns
            )
            if score > 0:
                scores[fault_type] = score

        if scores:
            return max(scores, key=scores.get)

        # Fallback: generic test failure
        if any(kw in log_lower for kw in ["fail", "error", "exception"]):
            return "LogicError"

        return "Unknown"

    def _extract_error_lines(self, log_text: str, max_lines: int = 50) -> List[str]:
        """Extract the most relevant error lines from a CI log."""
        lines = log_text.split("\n")
        error_lines = []
        context_window = 3  # Lines before/after error

        for i, line in enumerate(lines):
            line_lower = line.lower()
            is_error = any(kw in line_lower for kw in [
                "error", "exception", "fail", "assert",
                "traceback", "caused by", "fatal",
            ])
            if is_error:
                start = max(0, i - context_window)
                end = min(len(lines), i + context_window + 1)
                for j in range(start, end):
                    if lines[j] not in error_lines:
                        error_lines.append(lines[j])

        # Truncate to max_lines
        if len(error_lines) > max_lines:
            error_lines = error_lines[:max_lines]

        return error_lines if error_lines else lines[-20:]  # Last 20 lines as fallback

    def get_failed_runs(self, repo: str, max_runs: int = 10) -> List[Dict]:
        """Get recent failed workflow runs for a repo."""
        url = f"{GITHUB_API}/repos/{repo}/actions/runs"
        params = {
            "status": "failure",
            "per_page": min(max_runs, 100),
        }
        data = self._api_get(url, params)
        if not data or "workflow_runs" not in data:
            return []
        return data["workflow_runs"][:max_runs]

    def get_successful_runs(self, repo: str, max_runs: int = 5) -> List[Dict]:
        """Get recent successful workflow runs (for negative samples)."""
        url = f"{GITHUB_API}/repos/{repo}/actions/runs"
        params = {
            "status": "success",
            "per_page": min(max_runs, 100),
        }
        data = self._api_get(url, params)
        if not data or "workflow_runs" not in data:
            return []
        return data["workflow_runs"][:max_runs]

    def get_run_logs(self, repo: str, run_id: int) -> Optional[str]:
        """Download workflow run logs."""
        # Get jobs for the run
        url = f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}/jobs"
        data = self._api_get(url)
        if not data or "jobs" not in data:
            return None

        # Find the failed job
        failed_jobs = [j for j in data["jobs"] if j["conclusion"] == "failure"]
        if not failed_jobs:
            # For successful runs, get any completed job
            failed_jobs = [j for j in data["jobs"] if j["conclusion"] == "success"]

        if not failed_jobs:
            return None

        job = failed_jobs[0]

        # Get job log
        log_url = f"{GITHUB_API}/repos/{repo}/actions/jobs/{job['id']}/logs"
        log_text = self._api_get_text(log_url)

        return log_text

    def get_commit_diff(self, repo: str, sha: str) -> Optional[Dict]:
        """Get commit diff and metadata."""
        url = f"{GITHUB_API}/repos/{repo}/commits/{sha}"
        data = self._api_get(url)
        if not data:
            return None

        # Check diff size
        files = data.get("files", [])
        if len(files) > MAX_DIFF_FILES:
            return None

        # Build unified diff
        diff_parts = []
        total_additions = 0
        total_deletions = 0
        files_changed = []

        for f in files:
            patch = f.get("patch", "")
            if patch:
                diff_parts.append(
                    f"--- a/{f['filename']}\n+++ b/{f['filename']}\n{patch}"
                )
            files_changed.append(f["filename"])
            total_additions += f.get("additions", 0)
            total_deletions += f.get("deletions", 0)

        diff = "\n".join(diff_parts)
        if len(diff) > MAX_DIFF_SIZE or len(diff.split("\n")) < MIN_DIFF_LINES:
            return None

        return {
            "diff": diff,
            "files_changed": files_changed,
            "additions": total_additions,
            "deletions": total_deletions,
            "message": data.get("commit", {}).get("message", ""),
            "author": data.get("commit", {}).get("author", {}).get("name", ""),
            "date": data.get("commit", {}).get("author", {}).get("date", ""),
        }

    def process_run(self, repo: str, run: Dict, label: int) -> Optional[CICodeLogPair]:
        """Process a single workflow run into a code-log pair."""
        run_id = run["id"]
        sha = run.get("head_sha", "")

        if not sha:
            return None

        # Check if already collected
        pair_id = f"gha_{repo.replace('/', '_')}_{sha[:8]}"
        if any(p.id == pair_id for p in self.collected):
            return None

        # Get commit diff
        commit_data = self.get_commit_diff(repo, sha)
        if not commit_data:
            return None

        # Get CI logs
        log_text = self.get_run_logs(repo, run_id)
        if not log_text:
            return None

        # Extract error lines and classify
        error_lines = self._extract_error_lines(log_text)
        fault_type = self._classify_fault_type(log_text) if label == 1 else "normal"

        # Detect language from file extensions
        exts = [Path(f).suffix for f in commit_data["files_changed"]]
        if ".java" in exts:
            language = "java"
        elif ".py" in exts:
            language = "python"
        else:
            language = "mixed"

        pair = CICodeLogPair(
            id=pair_id,
            repo=repo,
            commit_sha=sha,
            commit_message=commit_data["message"][:200],
            code_change={
                "diff": commit_data["diff"],
                "files_changed": commit_data["files_changed"],
                "additions": commit_data["additions"],
                "deletions": commit_data["deletions"],
                "project": repo.split("/")[-1],
            },
            log_context={
                "messages": error_lines,
                "source": "github_actions",
                "workflow_name": run.get("name", ""),
                "job_name": run.get("display_title", ""),
                "full_log_hash": hashlib.md5(
                    log_text.encode()
                ).hexdigest(),
            },
            fault_type=fault_type,
            label=label,
            metadata={
                "run_id": run_id,
                "run_url": run.get("html_url", ""),
                "created_at": run.get("created_at", ""),
                "language": language,
                "log_length": len(log_text),
                "diff_length": len(commit_data["diff"]),
            },
        )
        return pair

    def collect_from_repo(
        self, repo: str, max_failures: int = 20, max_successes: int = 10
    ) -> Tuple[int, int]:
        """Collect code-log pairs from a single repository."""
        failures_collected = 0
        successes_collected = 0

        # Collect failed runs (positive samples)
        failed_runs = self.get_failed_runs(repo, max_runs=max_failures * 2)
        for run in failed_runs:
            if failures_collected >= max_failures:
                break
            pair = self.process_run(repo, run, label=1)
            if pair:
                self.collected.append(pair)
                failures_collected += 1

        # Collect successful runs (negative samples)
        success_runs = self.get_successful_runs(repo, max_runs=max_successes * 2)
        for run in success_runs:
            if successes_collected >= max_successes:
                break
            pair = self.process_run(repo, run, label=0)
            if pair:
                self.collected.append(pair)
                successes_collected += 1

        return failures_collected, successes_collected

    def collect_curated(self, target_pairs: int = 100):
        """Collect from curated repo list."""
        print(f"Collecting real CI/CD code-log pairs from {len(CURATED_REPOS)} repos")
        print(f"Target: {target_pairs} pairs")
        print("=" * 60)

        pairs_per_repo = max(3, target_pairs // len(CURATED_REPOS) + 1)

        for repo in tqdm(CURATED_REPOS, desc="Repos"):
            if len(self.collected) >= target_pairs:
                break

            tqdm.write(f"  Processing {repo}...")
            fail_count, success_count = self.collect_from_repo(
                repo,
                max_failures=pairs_per_repo,
                max_successes=max(1, pairs_per_repo // 3),
            )
            tqdm.write(
                f"    Collected: {fail_count} failures, {success_count} successes "
                f"(total: {len(self.collected)})"
            )

            # Save checkpoint periodically
            if len(self.collected) % 20 == 0:
                self._save_checkpoint()

        self._save_final()

    def collect_search(
        self, language: str = "java", max_repos: int = 50, target_pairs: int = 100
    ):
        """Search for repos and collect."""
        print(f"Searching for {language} repos with GitHub Actions...")

        url = f"{GITHUB_API}/search/repositories"
        params = {
            "q": f"language:{language} stars:>500 pushed:>2025-01-01",
            "sort": "stars",
            "per_page": min(max_repos, 100),
        }
        data = self._api_get(url, params)
        if not data or "items" not in data:
            print("ERROR: Failed to search repositories")
            return

        repos = [item["full_name"] for item in data["items"][:max_repos]]
        print(f"Found {len(repos)} repos")

        for repo in tqdm(repos, desc="Repos"):
            if len(self.collected) >= target_pairs:
                break

            tqdm.write(f"  Processing {repo}...")
            fail_count, success_count = self.collect_from_repo(repo)
            tqdm.write(
                f"    Collected: {fail_count} failures, {success_count} successes"
            )

        self._save_final()

    def _checkpoint_path(self) -> Path:
        return self.output_dir / "collection_checkpoint.json"

    def _load_checkpoint(self):
        """Load checkpoint from previous run."""
        cp = self._checkpoint_path()
        if cp.exists():
            with open(cp) as f:
                data = json.load(f)
            self.collected = [CICodeLogPair(**item) for item in data]
            print(f"Resumed from checkpoint: {len(self.collected)} pairs")

    def _save_checkpoint(self):
        """Save intermediate checkpoint."""
        with open(self._checkpoint_path(), "w") as f:
            json.dump([asdict(p) for p in self.collected], f, indent=2)

    def _save_final(self):
        """Save final dataset."""
        # Save full dataset
        output_file = self.output_dir / "real_ci_pairs.json"
        with open(output_file, "w") as f:
            json.dump([asdict(p) for p in self.collected], f, indent=2)

        # Save statistics
        stats = self._compute_stats()
        stats_file = self.output_dir / "collection_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        # Save in CodeFault evaluation format
        eval_file = self.output_dir / "real_ci_eval.json"
        eval_data = self._convert_to_eval_format()
        with open(eval_file, "w") as f:
            json.dump(eval_data, f, indent=2)

        print("\n" + "=" * 60)
        print(f"Collection complete!")
        print(f"  Total pairs: {len(self.collected)}")
        print(f"  Positive (fault): {stats['positive_count']}")
        print(f"  Negative (benign): {stats['negative_count']}")
        print(f"  Repos covered: {stats['repo_count']}")
        print(f"  Fault types: {stats['fault_type_distribution']}")
        print(f"\nOutput files:")
        print(f"  Full dataset:  {output_file}")
        print(f"  Eval format:   {eval_file}")
        print(f"  Statistics:    {stats_file}")

        # Remove checkpoint
        cp = self._checkpoint_path()
        if cp.exists():
            cp.unlink()

    def _compute_stats(self) -> Dict:
        """Compute dataset statistics."""
        fault_types = {}
        repos = set()
        languages = {}
        pos_count = 0
        neg_count = 0

        for p in self.collected:
            repos.add(p.repo)
            ft = p.fault_type
            fault_types[ft] = fault_types.get(ft, 0) + 1
            lang = p.metadata.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
            if p.label == 1:
                pos_count += 1
            else:
                neg_count += 1

        return {
            "total_pairs": len(self.collected),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "repo_count": len(repos),
            "repos": sorted(repos),
            "fault_type_distribution": dict(sorted(
                fault_types.items(), key=lambda x: -x[1]
            )),
            "language_distribution": languages,
            "collection_date": datetime.now().isoformat(),
            "mean_diff_length": (
                sum(len(p.code_change["diff"]) for p in self.collected)
                / max(len(self.collected), 1)
            ),
            "mean_log_lines": (
                sum(len(p.log_context["messages"]) for p in self.collected)
                / max(len(self.collected), 1)
            ),
        }

    def _convert_to_eval_format(self) -> List[Dict]:
        """Convert to CodeFault evaluation format (matching train.json schema)."""
        eval_samples = []
        for p in self.collected:
            eval_samples.append({
                "id": p.id,
                "code_change": {
                    "project": p.code_change["project"],
                    "commit_sha": p.commit_sha,
                    "diff": p.code_change["diff"],
                    "files_changed": p.code_change["files_changed"],
                    "additions": p.code_change["additions"],
                    "deletions": p.code_change["deletions"],
                },
                "log_context": {
                    "messages": p.log_context["messages"],
                    "source": "github_actions",
                },
                "label": p.label,
                "source": "github_actions",
                "fault_type": p.fault_type,
            })
        return eval_samples


def main():
    parser = argparse.ArgumentParser(
        description="Collect real CI/CD code-log pairs from GitHub Actions"
    )
    parser.add_argument(
        "--mode", choices=["curated", "search"], default="curated",
        help="Collection mode: 'curated' uses known repos, 'search' finds new ones"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent.parent / "data" / "real_ci",
        help="Output directory"
    )
    parser.add_argument(
        "--target", type=int, default=2000,
        help="Target number of pairs to collect (default 2000 for paper)"
    )
    parser.add_argument(
        "--language", default="java",
        help="Language filter for search mode"
    )
    parser.add_argument(
        "--max-repos", type=int, default=50,
        help="Maximum repos to search (search mode)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--token", default=None,
        help="GitHub token (default: GITHUB_TOKEN env var)"
    )

    args = parser.parse_args()

    # Get GitHub token
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GitHub token required.")
        print("Set GITHUB_TOKEN environment variable or pass --token")
        print("\nCreate a token at: https://github.com/settings/tokens")
        print("Required scopes: public_repo (for public repos)")
        sys.exit(1)

    # Clean checkpoint if not resuming
    if not args.resume:
        cp = args.output / "collection_checkpoint.json"
        if cp.exists():
            cp.unlink()

    collector = GitHubActionsCollector(token, args.output)

    if args.mode == "curated":
        collector.collect_curated(target_pairs=args.target)
    else:
        collector.collect_search(
            language=args.language,
            max_repos=args.max_repos,
            target_pairs=args.target,
        )


if __name__ == "__main__":
    main()
