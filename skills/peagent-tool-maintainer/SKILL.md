---
name: peagent-tool-maintainer
description: Maintain the peagent_tool Python package and peagent-tool CLI. Use when adding or modifying peagent_tool features, fixing bugs, changing public Python APIs or CLI commands, updating prediction/deployment behavior, adding tests, writing ReadTheDocs tutorials, or synchronizing confirmed peagent_tool changes to the YAOJ-bioin/peagent_tool GitHub main branch.
---

# PEAgent Tool Maintainer

## Operating Contract

Work in `/scratch/jy16611/peagent`. Keep the package surfaces aligned:

- Python import: `peagent_tool`
- pip distribution: `peagent-tool`
- CLI command: `peagent-tool`
- GitHub remote: `git@github.com:YAOJ-bioin/peagent_tool.git`
- Documentation site: `https://peagent-tool.readthedocs.io`

Read `references/repo-contract.md` before changing package behavior, runtime paths, or deployment instructions.

## Update Workflow

1. Start with repository hygiene:
   - Run `scripts/check_ready.py --repo /scratch/jy16611/peagent`.
   - If the repo is dirty, inspect changes and protect user work. Commit only files related to the current request.
   - Work on `main` tracking `origin/main`.
2. Implement the requested change conservatively:
   - Preserve simple imports and one-call helpers when possible.
   - Update CLI, README, docs, and tests whenever public behavior changes.
   - Do not commit model weights, metadata dumps, genomes, caches, environments, or cloud runtime assets.
3. For each new feature, add a tutorial:
   - Read `references/docs-workflow.md`.
   - Use `scripts/scaffold_tutorial.py --feature <feature-slug> --docs docs` as the starting point.
   - Include Python API usage, CLI usage, example output, and at least one visual element.
4. Validate:
   - `PYTHONPATH=src pytest -q`
   - `ruff check`
   - `python -m compileall -q src/peagent_tool`
   - `mkdocs build --strict` when MkDocs configuration exists or docs were touched.
5. Synchronize confirmed updates:
   - Read `references/release-sync.md`.
   - Re-run `scripts/check_ready.py --repo /scratch/jy16611/peagent --allow-dirty` before staging.
   - Stage only current-task files, commit with a clear message, then `git push origin main`.
   - Stop and report if tests fail, docs fail, unrelated files are dirty, or push is rejected.

## Runtime And HPC Guardrails

If the task touches model files, metadata transfer, cloud layout, `/opt/peagent/backend`, large files, Slurm, or PEAgent_S2E assets, also use `peagent-hpc-guardrails`. Never read or copy large runtime files unless that workflow allows it.

## Bundled Resources

- `scripts/check_ready.py`: preflight check for branch, remote, dirty state, staged risky assets, and large staged files.
- `scripts/scaffold_tutorial.py`: create or preview MkDocs tutorial files for a new feature.
- `references/repo-contract.md`: package and deployment invariants.
- `references/docs-workflow.md`: MkDocs Material and ReadTheDocs tutorial rules.
- `references/release-sync.md`: direct-to-main commit and push policy.
