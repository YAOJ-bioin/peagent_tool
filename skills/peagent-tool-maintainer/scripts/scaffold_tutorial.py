#!/usr/bin/env python3
"""Create a MkDocs tutorial scaffold for a peagent_tool feature."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        raise ValueError("feature must contain at least one letter or digit")
    return slug


def titleize(slug: str) -> str:
    keep_lower = {"api", "cli", "dna", "rna"}
    words = []
    for word in slug.split("-"):
        words.append(word.upper() if word in keep_lower else word.capitalize())
    return " ".join(words)


def tutorial_template(slug: str) -> str:
    title = titleize(slug)
    return f"""# {title}

## Purpose

Describe what this feature does and when a PEAgent Tool user should use it.

## Environment

```bash
pip install "git+ssh://git@github.com/YAOJ-bioin/peagent_tool.git@main"
export PEAGENT_BACKEND_ROOT=/opt/peagent/backend
```

## Workflow

```mermaid
flowchart LR
    A[Input sequence] --> B[peagent_tool API or CLI]
    B --> C[Load model and metadata]
    C --> D[Predict tissue_celltype scores]
    D --> E[Ranked output table]
```

## Python API

```python
from peagent_tool import predict_celltypes

df = predict_celltypes(
    "ACGT" * 336,
    species="soybean",
    mode="fast",
    top_k=5,
)
print(df)
```

## CLI

```bash
peagent-tool predict-celltypes \\
  --species soybean \\
  --sequence "$(python -c 'print(\"ACGT\" * 336)')" \\
  --mode fast \\
  --top-k 5
```

## Example Output

| rank | tissue_celltype | prediction |
| ---: | --- | ---: |
| 1 | Cotyledon_stage_seeds_SC_xylem | 0.0139 |
| 2 | Early_nodule_Nodule_infected_cell | 0.0103 |

## Notes

Add feature-specific constraints, expected input length behavior, and troubleshooting notes.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True, help="Feature name or slug.")
    parser.add_argument("--docs", type=Path, default=Path("docs"), help="Path to the MkDocs docs directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing tutorial.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned files without writing.")
    args = parser.parse_args(argv)

    slug = slugify(args.feature)
    docs_dir = args.docs.resolve()
    tutorial = docs_dir / "tutorials" / f"{slug}.md"
    assets_dir = docs_dir / "assets" / "tutorials" / slug
    gitkeep = assets_dir / ".gitkeep"
    content = tutorial_template(slug)

    planned = [tutorial, gitkeep]
    if args.dry_run:
        print("DRY RUN: would create tutorial scaffold")
        for path in planned:
            print(path)
        print("\n--- tutorial preview ---")
        print(content)
        return 0

    if tutorial.exists() and not args.force:
        raise SystemExit(f"ERROR: tutorial already exists: {tutorial}. Use --force to overwrite.")

    tutorial.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    tutorial.write_text(content, encoding="utf-8")
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")

    print(f"created {tutorial}")
    print(f"created {gitkeep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
