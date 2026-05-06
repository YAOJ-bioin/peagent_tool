"""Cell-type-level prediction for scBasset PEAgent models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .sequence import INPUT_LEN, SequenceRecord, onehot_from_normalized_sequence, parse_sequences


BOTTLENECK_SIZE = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKEND_ROOT = Path(os.environ.get("PEAGENT_BACKEND_ROOT", "/opt/peagent/backend"))
DEFAULT_MODEL_DIR = DEFAULT_BACKEND_ROOT / "models"
DEFAULT_METADATA_DIR = DEFAULT_BACKEND_ROOT / "metadata"


@dataclass(frozen=True)
class SpeciesConfig:
    """Default model metadata needed to aggregate cell-level predictions."""

    id: str
    display_name: str
    n_cells: int
    model_path: Path
    metadata_path: Path
    bottleneck_size: int = BOTTLENECK_SIZE


SPECIES_CONFIGS = {
    "soybean": SpeciesConfig(
        id="soybean",
        display_name="Soybean",
        n_cells=200732,
        model_path=DEFAULT_MODEL_DIR / "soybean" / "model.h5",
        metadata_path=DEFAULT_METADATA_DIR / "soybean_group_metadata.csv",
    ),
    "maize": SpeciesConfig(
        id="maize",
        display_name="Maize",
        n_cells=50639,
        model_path=DEFAULT_MODEL_DIR / "maize" / "model.h5",
        metadata_path=DEFAULT_METADATA_DIR / "maize_group_metadata.csv",
    ),
    "rice": SpeciesConfig(
        id="rice",
        display_name="Rice",
        n_cells=104029,
        model_path=DEFAULT_MODEL_DIR / "rice" / "model.h5",
        metadata_path=DEFAULT_METADATA_DIR / "rice_group_metadata.csv",
    ),
}


def _normalize_species(species: str) -> str:
    key = str(species).strip().lower()
    aliases = {
        "glycine max": "soybean",
        "soy": "soybean",
        "zea mays": "maize",
        "corn": "maize",
        "oryza sativa": "rice",
    }
    return aliases.get(key, key)


def resolve_species_config(
    species: str,
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
) -> SpeciesConfig:
    """Resolve a species config with optional model and metadata overrides."""

    key = _normalize_species(species)
    if key not in SPECIES_CONFIGS:
        raise KeyError("Unknown species: %s. Supported species: %s" % (species, ", ".join(sorted(SPECIES_CONFIGS))))
    cfg = SPECIES_CONFIGS[key]
    return replace(
        cfg,
        model_path=Path(model_path) if model_path is not None else cfg.model_path,
        metadata_path=Path(metadata_path) if metadata_path is not None else cfg.metadata_path,
        n_cells=int(n_cells) if n_cells is not None else cfg.n_cells,
        bottleneck_size=int(bottleneck_size),
    )


def _load_group_metadata(path: Union[str, Path], n_cells: int) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Missing group metadata file: %s" % path)

    df = pd.read_csv(path)
    required = {"tissue_celltype", "tissue", "celltype", "n_cells", "group_idx", "cell_indices"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Group metadata %s is missing columns: %s" % (path, ", ".join(sorted(missing))))

    df = df.sort_values("group_idx").reset_index(drop=True).copy()
    df["group_idx"] = df["group_idx"].astype(np.int32)
    df["n_cells"] = df["n_cells"].astype(np.int32)

    def parse_cell_indices(value):
        if isinstance(value, str):
            parsed = json.loads(value)
        else:
            parsed = value
        arr = np.asarray(parsed, dtype=np.int32)
        if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= int(n_cells)):
            raise ValueError("cell_indices in %s exceed n_cells=%d" % (path, n_cells))
        return arr

    df["cell_indices"] = df["cell_indices"].apply(parse_cell_indices)
    if df["tissue_celltype"].duplicated().any():
        dup = df.loc[df["tissue_celltype"].duplicated(), "tissue_celltype"].iloc[0]
        raise ValueError("Duplicate tissue_celltype in metadata: %s" % dup)
    return df


def _build_group_weight_matrix(group_df: pd.DataFrame, n_cells: int) -> np.ndarray:
    weights = np.zeros((int(n_cells), len(group_df)), dtype=np.float32)
    for row in group_df.itertuples(index=False):
        idx = np.asarray(row.cell_indices, dtype=np.int32)
        if idx.size:
            weights[idx, int(row.group_idx)] = 1.0 / float(row.n_cells)
    return weights


def _aggregate_dense_parameters(
    kernel: np.ndarray,
    bias: np.ndarray,
    group_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate final cell-level logit parameters to tissue-celltype logits."""

    kernel = np.asarray(kernel, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    group_kernel = np.empty((kernel.shape[0], len(group_df)), dtype=np.float32)
    group_bias = np.empty((len(group_df),), dtype=np.float32)
    for row in group_df.itertuples(index=False):
        idx = np.asarray(row.cell_indices, dtype=np.int32)
        if idx.size == 0:
            group_kernel[:, int(row.group_idx)] = 0.0
            group_bias[int(row.group_idx)] = 0.0
        else:
            group_kernel[:, int(row.group_idx)] = kernel[:, idx].mean(axis=1)
            group_bias[int(row.group_idx)] = bias[idx].mean(axis=0)
    return group_kernel, group_bias


def _get_make_model():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from ._vendor.scbasset_model import make_model

    return make_model


class CellTypePredictor:
    """Predict valid tissue-celltype scores from DNA sequence input."""

    def __init__(
        self,
        species: str,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        n_cells: Optional[int] = None,
        bottleneck_size: int = BOTTLENECK_SIZE,
        mode: str = "exact",
        model=None,
    ):
        self.config = resolve_species_config(
            species=species,
            model_path=model_path,
            metadata_path=metadata_path,
            n_cells=n_cells,
            bottleneck_size=bottleneck_size,
        )
        self.mode = self._normalize_mode(mode)
        self._provided_model = model

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        value = str(mode).strip().lower()
        if value not in {"exact", "fast"}:
            raise ValueError("mode must be 'exact' or 'fast', got %r" % mode)
        return value

    @cached_property
    def group_metadata(self) -> pd.DataFrame:
        return _load_group_metadata(self.config.metadata_path, self.config.n_cells)

    @cached_property
    def group_weights(self) -> np.ndarray:
        return _build_group_weight_matrix(self.group_metadata, self.config.n_cells)

    @cached_property
    def valid_celltypes(self) -> Sequence[str]:
        return self.group_metadata["tissue_celltype"].astype(str).tolist()

    @cached_property
    def model(self):
        if self._provided_model is not None:
            return self._provided_model
        if self.mode == "fast":
            return self._build_fast_model()
        return self._build_exact_model()

    def _build_exact_model(self):
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError("Missing model file: %s" % model_path)
        make_model = _get_make_model()
        model = make_model(
            self.config.bottleneck_size,
            int(self.config.n_cells),
            seq_len=INPUT_LEN,
            show_summary=False,
        )
        model.load_weights(str(model_path))
        return model

    def _build_fast_model(self):
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError("Missing model file: %s" % model_path)

        make_model = _get_make_model()
        import tensorflow as tf

        full_model = make_model(
            self.config.bottleneck_size,
            int(self.config.n_cells),
            seq_len=INPUT_LEN,
            show_summary=False,
        )
        full_model.load_weights(str(model_path))

        final_dense = self._find_final_dense_layer(full_model, tf)
        kernel, bias = final_dense.get_weights()
        group_kernel, group_bias = _aggregate_dense_parameters(kernel, bias, self.group_metadata)

        group_dense = tf.keras.layers.Dense(
            units=len(self.group_metadata),
            use_bias=True,
            activation="sigmoid",
            name="fast_tissue_celltype_prediction",
        )
        group_output = group_dense(final_dense.input)
        group_output = tf.keras.layers.Flatten(name="fast_tissue_celltype_flatten")(group_output)
        fast_model = tf.keras.Model(inputs=full_model.input, outputs=group_output)
        group_dense.set_weights([group_kernel, group_bias])
        return fast_model

    def _find_final_dense_layer(self, model, tf):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                if len(weights) == 2 and weights[0].shape[1] == int(self.config.n_cells):
                    return layer
        raise ValueError("Could not find final Dense layer with %d outputs." % self.config.n_cells)

    def predict(
        self,
        sequences: Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]],
        top_k: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> pd.DataFrame:
        """Return long, ranked tissue-celltype predictions."""

        records = parse_sequences(sequences, input_len=INPUT_LEN)
        if top_k is not None and int(top_k) <= 0:
            raise ValueError("top_k must be positive when provided.")

        scores_by_id = self._predict_matrix(records, batch_size=batch_size)
        return self._format_long(scores_by_id, records, top_k=top_k)

    def predict_wide(
        self,
        sequences: Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> pd.DataFrame:
        """Return a sequence-by-tissue-celltype prediction matrix."""

        records = parse_sequences(sequences, input_len=INPUT_LEN)
        scores = self._predict_matrix(records, batch_size=batch_size)
        out = pd.DataFrame(scores, columns=self.valid_celltypes)
        out.insert(0, "sequence_id", [record.sequence_id for record in records])
        return out

    def _predict_matrix(self, records: Sequence[SequenceRecord], batch_size: int) -> np.ndarray:
        outputs = []
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            x = np.stack(
                [
                    onehot_from_normalized_sequence(record.normalized_sequence, input_len=INPUT_LEN)
                    for record in batch_records
                ],
                axis=0,
            )
            pred = self.model.predict_on_batch(x).astype(np.float32, copy=False)
            if self.mode == "exact":
                pred = pred @ self.group_weights
            outputs.append(pred.astype(np.float32, copy=False))
        if not outputs:
            return np.empty((0, len(self.group_metadata)), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def _format_long(self, scores: np.ndarray, records: Sequence[SequenceRecord], top_k: Optional[int]) -> pd.DataFrame:
        meta = self.group_metadata[["tissue_celltype", "tissue", "celltype", "n_cells"]].copy()
        frames = []
        for i, record in enumerate(records):
            df = meta.copy()
            df.insert(0, "prediction", scores[i].astype(np.float32, copy=False))
            df = df.sort_values("prediction", ascending=False).reset_index(drop=True)
            df.insert(0, "rank", np.arange(1, len(df) + 1, dtype=np.int32))
            if top_k is not None:
                df = df.head(int(top_k)).copy()
            df.insert(0, "sequence_id", record.sequence_id)
            df["original_length"] = int(record.original_length)
            df["normalized_length"] = int(record.normalized_length)
            df["normalization_action"] = record.normalization_action
            df["species"] = self.config.id
            df["mode"] = self.mode
            df["model_path"] = str(self.config.model_path)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def load_predictor(
    species: str,
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    mode: str = "exact",
) -> CellTypePredictor:
    """Load a reusable cell-type predictor."""

    return CellTypePredictor(
        species=species,
        model_path=model_path,
        metadata_path=metadata_path,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
        mode=mode,
    )


def predict_celltypes(
    sequences: Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]],
    species: str,
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    mode: str = "exact",
    top_k: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Convenience wrapper for one-off tissue-celltype prediction."""

    predictor = load_predictor(
        species=species,
        model_path=model_path,
        metadata_path=metadata_path,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
        mode=mode,
    )
    return predictor.predict(sequences, top_k=top_k, batch_size=batch_size)


def list_valid_celltypes(
    species: str,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
) -> pd.DataFrame:
    """List valid tissue-celltype outputs for a species."""

    cfg = resolve_species_config(species=species, metadata_path=metadata_path, n_cells=n_cells)
    df = _load_group_metadata(cfg.metadata_path, cfg.n_cells)
    return df[["group_idx", "tissue_celltype", "tissue", "celltype", "n_cells"]].copy()
