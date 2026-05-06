# ISM Attribution

## Purpose

ISM attribution highlights which reference bases in one DNA sequence move a
PEAgent/scBasset target up or down under in silico mutagenesis. Use it after you
have selected a species model and a target: either `global` or one exact valid
`tissue_celltype`.

## Environment

```bash
pip install "peagent-tool[ism] @ git+ssh://git@github.com/YAOJ-bioin/peagent_tool.git@main"
export PEAGENT_BACKEND_ROOT=/opt/peagent/backend
```

The backend root must contain the same runtime assets used for cell-type
prediction:

```text
/opt/peagent/backend/models/<species>/model.h5
/opt/peagent/backend/metadata/<species>_group_metadata.csv
```

## Workflow

```mermaid
flowchart LR
    A[Reference sequence] --> B[Normalize to 1344 bp]
    B --> C[Mutate each position to A/C/G/T]
    C --> D[Compute bottleneck deltas]
    D --> E[Project to global or tissue_celltype weights]
    E --> F[Center across bases]
    F --> G[Draw ref-base logo as PDF and PNG]
```

## Python API

```python
from peagent_tool import compute_and_plot_ism_attribution

result, paths = compute_and_plot_ism_attribution(
    "ACGT" * 336,
    species="maize",
    target="global",
    out_prefix="outputs/maize_global_ism",
    plot_start=600,
    plot_end=690,
)
print(paths["pdf"])
```

## CLI

```bash
peagent-tool ism-attribution \
  --species maize \
  --sequence "$(python -c 'print("ACGT" * 336)')" \
  --target global \
  --plot-start 600 \
  --plot-end 690 \
  --out-prefix outputs/maize_global_ism
```

For a cell-type-specific map, first list the valid names:

```bash
peagent-tool list-celltypes --species maize
```

Then pass the exact `tissue_celltype` value:

```bash
peagent-tool ism-attribution \
  --species maize \
  --fasta ref_sequence.fa \
  --target Emb_epidermis \
  --plot-start 600 \
  --plot-end 690 \
  --out-prefix outputs/maize_emb_epidermis_ism
```

## Example Output

The CLI prints the generated files:

```text
sequence_id	seq_1
species	maize
target	global
pdf	outputs/maize_global_ism.pdf
png	outputs/maize_global_ism.png
font	Liberation Sans
```

The PDF is vector output for editing. The PNG is exported at 600 dpi.

```text
Position:      600            620            640            660            680
Attribution:   A   C   G   T sequence-logo letters above or below zero
```

## Notes

- `--plot-start` and `--plot-end` are 1-based positions on the normalized
  1344 bp model input.
- `--position-offset` shifts the x-axis labels without changing the sequence
  slice.
- `N` bases are allowed in the model input but are drawn as zero-height letters
  in the ref-base attribution logo.
- This feature does not subtract tissue averages and does not require genomes,
  annotations, or BigWig output.
