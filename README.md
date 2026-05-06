# PEAgent Tool

![PyPI version](https://img.shields.io/pypi/v/peagent-tool.svg)
[![Documentation Status](https://readthedocs.org/projects/peagent-tool/badge/?version=latest)](https://peagent-tool.readthedocs.io/en/latest/?version=latest)

Plant Epigenomics prediction utilities for PEAgent.

* PyPI package: https://pypi.org/project/peagent-tool/
* Free software: MIT License
* Documentation: https://peagent-tool.readthedocs.io.

## Features

* Predict valid tissue-celltype scores from DNA sequence input using PEAgent/scBasset models.
* Supports two prediction modes:
  * `exact`: predicts every model cell output and averages probabilities within each valid `tissue_celltype`.
  * `fast`: aggregates the final cell-level Dense logits to `tissue_celltype` outputs before sigmoid; this is faster at prediction time but approximate.

## Usage

```python
from peagent_tool import predict_celltypes

scores = predict_celltypes(
    "ACGT" * 336,
    species="soybean",
    mode="exact",
    top_k=20,
)
```

By default, runtime assets are resolved under `/opt/peagent/backend`:

```text
/opt/peagent/backend/models/<species>/model.h5
/opt/peagent/backend/metadata/<species>_group_metadata.csv
```

Set `PEAGENT_BACKEND_ROOT` to use a different runtime asset directory.

For repeated calls, load the model once:

```python
from peagent_tool import load_predictor

predictor = load_predictor("maize", mode="fast")
scores = predictor.predict({"seq1": "ACGT" * 336, "seq2": "TGCA" * 336}, top_k=10)
```

CLI:

```bash
peagent-tool predict-celltypes --species soybean --sequence "$(python - <<'PY'
print("ACGT" * 336)
PY
)" --mode exact --top-k 20
```

Run in an environment that can import TensorFlow. The package vendors the minimal
scBasset architecture needed to load the existing PEAgent `.h5` weights, so a
full scBasset checkout is not required for inference.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
