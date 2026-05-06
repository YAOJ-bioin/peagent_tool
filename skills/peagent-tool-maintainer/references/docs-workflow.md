# MkDocs Tutorial Workflow

## Documentation Stack

Use MkDocs Material for ReadTheDocs. If the repository does not yet have docs build configuration and the current task changes docs or adds a feature tutorial, add:

- `mkdocs.yml`
- `.readthedocs.yaml`
- docs dependencies in the project configuration or a dedicated docs requirements file

The docs must build with:

```bash
mkdocs build --strict
```

## Tutorial Requirement

Every new user-facing feature needs a tutorial under:

```text
docs/tutorials/<feature-slug>.md
docs/assets/tutorials/<feature-slug>/
```

Start with:

```bash
skills/peagent-tool-maintainer/scripts/scaffold_tutorial.py --feature <feature-slug> --docs docs
```

## Tutorial Content

Each tutorial must include:

- Purpose and when to use the feature.
- Installation or environment assumptions.
- Python API example.
- CLI example.
- Example output as a table or fenced text block.
- A visual element: Mermaid flowchart, screenshot, or generated lightweight PNG/SVG.

Prefer Mermaid for process diagrams because it is versionable and lightweight. Do not add runtime models, genomes, full metadata, or large binary assets to docs.

## Navigation

Ensure the tutorial is reachable from the MkDocs navigation. Keep tutorial names user-facing and feature-specific, for example `Predict Cell Types From Sequence`.

When replacing legacy Markdown docs, preserve the important install, CLI, runtime asset, and API examples.
