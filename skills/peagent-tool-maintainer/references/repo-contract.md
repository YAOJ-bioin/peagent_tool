# PEAgent Tool Repo Contract

## Fixed Surfaces

- Repository: `/scratch/jy16611/peagent`
- GitHub: `YAOJ-bioin/peagent_tool`
- Remote URL: `git@github.com:YAOJ-bioin/peagent_tool.git`
- Branch: `main`
- Python package import: `peagent_tool`
- Distribution name: `peagent-tool`
- CLI: `peagent-tool`

Keep these names stable unless the user explicitly requests a rename.

## Runtime Assets

Cloud runtime assets live outside the package under `/opt/peagent/backend`:

```text
/opt/peagent/backend/
├── models/<species>/model.h5
├── metadata/<species>_group_metadata.csv
├── envs/peagent-pred/
├── scripts/
├── jobs/
└── logs/
```

Do not commit model weights, metadata dumps, genomes, FASTA/FASTQ/BAM-like data, conda envs, caches, or `/opt/peagent/backend` contents.

## Prediction Package Defaults

Default runtime root is `PEAGENT_BACKEND_ROOT` or `/opt/peagent/backend`.

Default paths:

```text
/opt/peagent/backend/models/<species>/model.h5
/opt/peagent/backend/metadata/<species>_group_metadata.csv
```

The package should keep sequence-string prediction simple through:

- `from peagent_tool import predict_celltypes`
- `from peagent_tool import load_predictor`
- `peagent-tool predict-celltypes ...`
- `peagent-tool list-celltypes ...`

## Standard Validation

Run the relevant subset before committing:

```bash
PYTHONPATH=src pytest -q
ruff check
python -m compileall -q src/peagent_tool
mkdocs build --strict
```

Use `mkdocs build --strict` when docs are touched or MkDocs configuration exists.
