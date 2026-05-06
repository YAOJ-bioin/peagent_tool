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
