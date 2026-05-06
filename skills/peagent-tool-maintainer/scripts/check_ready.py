#!/usr/bin/env python3
"""Preflight checks for peagent_tool maintenance work."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

EXPECTED_REMOTE = "git@github.com:YAOJ-bioin/peagent_tool.git"
FORBIDDEN_PATTERNS = (
    "*.h5",
    "*.hdf5",
    "*.pt",
    "*.pth",
    "*.pkl",
    "*.pickle",
    "*.npy",
    "*.npz",
    "*.h5ad",
    "*.fa",
    "*.fasta",
    "*.fastq",
    "*.fq",
    "*.bam",
    "*.sam",
    "*.cram",
    "models/*",
    "metadata/*",
    "data/*",
    "envs/*",
    "jobs/*",
    "logs/*",
    "site/*",
    "docs/_build/*",
)


def run_git(repo: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc


def git_lines(repo: Path, args: list[str]) -> list[str]:
    out = run_git(repo, args).stdout
    return [line for line in out.splitlines() if line]


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_forbidden(path: str) -> bool:
    normalized = normalize_path(path)
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in FORBIDDEN_PATTERNS)


def staged_files(repo: Path) -> list[str]:
    out = run_git(repo, ["diff", "--cached", "--name-only", "-z"]).stdout
    return [item for item in out.split("\0") if item]


def staged_size(repo: Path, path: str) -> int:
    proc = run_git(repo, ["cat-file", "-s", f":{path}"], check=False)
    if proc.returncode != 0:
        return 0
    return int(proc.stdout.strip() or "0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/scratch/jy16611/peagent"))
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty worktree but still inspect staged files.",
    )
    parser.add_argument("--max-staged-mb", type=float, default=10.0, help="Maximum allowed size for one staged file.")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        print(f"ERROR: not a git repository: {repo}", file=sys.stderr)
        return 2

    errors: list[str] = []
    branch = run_git(repo, ["branch", "--show-current"]).stdout.strip()
    if branch != "main":
        errors.append(f"current branch is {branch!r}, expected 'main'")

    origin_push = run_git(repo, ["remote", "get-url", "--push", "origin"], check=False).stdout.strip()
    if origin_push != EXPECTED_REMOTE:
        errors.append(f"origin push URL is {origin_push!r}, expected {EXPECTED_REMOTE!r}")

    upstream = run_git(repo, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False)
    if upstream.returncode != 0 or upstream.stdout.strip() != "origin/main":
        errors.append("main is not tracking origin/main")

    status = git_lines(repo, ["status", "--porcelain=v1"])
    if status and not args.allow_dirty:
        errors.append("worktree is dirty; inspect user changes or rerun with --allow-dirty before staging")

    dirty_paths = [line[3:] for line in status if len(line) > 3]
    forbidden_dirty = sorted(path for path in dirty_paths if is_forbidden(path))
    if forbidden_dirty:
        errors.append("dirty forbidden runtime/data files: " + ", ".join(forbidden_dirty))

    max_bytes = int(args.max_staged_mb * 1024 * 1024)
    for path in staged_files(repo):
        if is_forbidden(path):
            errors.append(f"staged forbidden runtime/data file: {path}")
        size = staged_size(repo, path)
        if size > max_bytes:
            errors.append(f"staged file exceeds {args.max_staged_mb:g} MB: {path} ({size} bytes)")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {repo} is ready for peagent_tool maintenance")
    print(f"branch={branch}")
    print(f"origin={origin_push}")
    if status:
        print(f"dirty_entries={len(status)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
