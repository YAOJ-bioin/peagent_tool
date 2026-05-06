# Usage

Predict valid `tissue_celltype` scores from a DNA sequence:

```python
from peagent_tool import predict_celltypes

df = predict_celltypes("ACGT" * 336, species="soybean", mode="exact", top_k=20)
```

Use `mode="exact"` for the canonical result: the model predicts all cells, then PEAgent averages
probabilities within each valid tissue-celltype group.

Use `mode="fast"` for a faster approximation: PEAgent averages the final Dense logits parameters
within each valid tissue-celltype group, then applies sigmoid directly to the grouped logits.

```python
from peagent_tool import load_predictor

predictor = load_predictor("rice", mode="fast")
df = predictor.predict({"seq1": "ACGT" * 336, "seq2": "TGCA" * 336}, top_k=10)
```

By default, model and metadata assets are read from `/opt/peagent/backend`.
Override this with `PEAGENT_BACKEND_ROOT` or pass `model_path` and
`metadata_path` explicitly.

To use PEAgent Tool in a project:

```python
import peagent_tool
```

## ISM attribution maps

Use ISM attribution when you want a sequence-logo style attribution map for one
reference sequence and one target. The target can be `global` or one exact valid
`tissue_celltype` shown by `peagent-tool list-celltypes`.

```python
from peagent_tool import compute_and_plot_ism_attribution

result, paths = compute_and_plot_ism_attribution(
    "ACGT" * 336,
    species="maize",
    target="global",
    out_prefix="maize_global_ism",
    plot_start=600,
    plot_end=690,
)
print(paths["pdf"])
```

```bash
peagent-tool ism-attribution \
  --species maize \
  --sequence "$(python -c 'print("ACGT" * 336)')" \
  --target global \
  --plot-start 600 \
  --plot-end 690 \
  --out-prefix maize_global_ism
```

The command writes `maize_global_ism.pdf` and `maize_global_ism.png`.
`--plot-start` and `--plot-end` are 1-based positions on the normalized 1344 bp
model input. `--position-offset` shifts the x-axis labels when you want to show
local sequence coordinates.
