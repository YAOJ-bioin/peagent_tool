"""In silico mutagenesis attribution maps for PEAgent scBasset models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .celltype_prediction import (
    BOTTLENECK_SIZE,
    _get_make_model,
    _load_group_metadata,
    resolve_species_config,
)
from .sequence import INPUT_LEN, SequenceRecord, onehot_from_normalized_sequence, parse_sequences

BASES = ("A", "C", "G", "T")
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
DEFAULT_POSITION_BATCH_SIZE = 32
DNAInput = Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]]


@dataclass(frozen=True)
class ISMAttributionResult:
    """Computed ISM attribution for one sequence and one target."""

    sequence_id: str
    species: str
    species_display_name: str
    target: str
    target_label: str
    tissue: Optional[str]
    celltype: Optional[str]
    n_cells: int
    original_length: int
    normalized_length: int
    normalization_action: str
    normalized_sequence: str
    centered_scores: np.ndarray
    attribution: np.ndarray
    model_path: str
    metadata_path: str

    def attribution_dataframe(self) -> pd.DataFrame:
        """Return ref-base attribution scores as an A/C/G/T matrix."""

        return pd.DataFrame(self.attribution, columns=BASES)

    def centered_scores_dataframe(self) -> pd.DataFrame:
        """Return centered hypothetical ISM scores as an A/C/G/T matrix."""

        return pd.DataFrame(self.centered_scores, columns=BASES)


def _strict_reference_onehot(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for pos, base in enumerate(seq.upper()):
        idx = BASE_TO_INDEX.get(base)
        if idx is not None:
            arr[pos, idx] = 1.0
    return arr


def _center_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("scores must have shape (sequence_length, 4).")
    return scores - scores.mean(axis=1, keepdims=True)


def _reference_attribution(centered_scores: np.ndarray, normalized_sequence: str) -> np.ndarray:
    centered_scores = np.asarray(centered_scores, dtype=np.float32)
    if centered_scores.shape != (len(normalized_sequence), 4):
        raise ValueError("centered_scores shape must match normalized_sequence length.")
    return centered_scores * _strict_reference_onehot(normalized_sequence)


def _as_bottleneck_matrix(value, expected_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError("Bottleneck output must include batch and feature dimensions.")
    batch_size = int(arr.shape[0])
    arr = arr.reshape(batch_size, -1)
    if arr.shape[1] != int(expected_size):
        raise ValueError("Expected bottleneck size %d, got %d." % (int(expected_size), int(arr.shape[1])))
    return arr


def _single_sequence_record(sequences: DNAInput) -> SequenceRecord:
    records = parse_sequences(sequences, input_len=INPUT_LEN)
    if len(records) != 1:
        raise ValueError("ISM attribution expects exactly one sequence per call, got %d." % len(records))
    return records[0]


def _mutation_batch(reference_onehot: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int32)
    batch = np.repeat(reference_onehot[None, :, :], repeats=len(positions) * 4, axis=0)
    eye = np.eye(4, dtype=np.float32)
    for offset, pos in enumerate(positions):
        batch[offset * 4 : offset * 4 + 4, int(pos), :] = eye
    return batch


def _find_final_dense_layer(model, tf, n_cells: int):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            if len(weights) == 2 and weights[0].shape[1] == int(n_cells):
                return layer
    raise ValueError("Could not find final Dense layer with %d outputs." % int(n_cells))


def _window_to_slice(sequence_length: int, plot_start: Optional[int], plot_end: Optional[int]) -> tuple[int, int]:
    start = 1 if plot_start is None else int(plot_start)
    end = int(sequence_length) if plot_end is None else int(plot_end)
    if start < 1:
        raise ValueError("plot_start must be >= 1.")
    if end < start:
        raise ValueError("plot_end must be >= plot_start.")
    if end > int(sequence_length):
        raise ValueError("plot_end must be <= normalized sequence length %d." % int(sequence_length))
    return start - 1, end


class ISMAttributor:
    """Compute one-sequence, one-target ISM attribution maps."""

    def __init__(
        self,
        species: str,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        n_cells: Optional[int] = None,
        bottleneck_size: int = BOTTLENECK_SIZE,
        model=None,
        bottleneck_model=None,
        dense_kernel: Optional[np.ndarray] = None,
    ):
        self.config = resolve_species_config(
            species=species,
            model_path=model_path,
            metadata_path=metadata_path,
            n_cells=n_cells,
            bottleneck_size=bottleneck_size,
        )
        self._provided_model = model
        self._provided_bottleneck_model = bottleneck_model
        self._provided_dense_kernel = dense_kernel

    @cached_property
    def group_metadata(self) -> pd.DataFrame:
        return _load_group_metadata(self.config.metadata_path, self.config.n_cells)

    @cached_property
    def valid_celltypes(self) -> Sequence[str]:
        return self.group_metadata["tissue_celltype"].astype(str).tolist()

    @cached_property
    def model(self):
        if self._provided_model is not None:
            return self._provided_model
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

    @cached_property
    def final_dense_layer(self):
        import tensorflow as tf

        return _find_final_dense_layer(self.model, tf, self.config.n_cells)

    @cached_property
    def dense_kernel(self) -> np.ndarray:
        if self._provided_dense_kernel is not None:
            kernel = np.asarray(self._provided_dense_kernel, dtype=np.float32)
        else:
            kernel = np.asarray(self.final_dense_layer.get_weights()[0], dtype=np.float32)
        if kernel.ndim != 2:
            raise ValueError("Final Dense kernel must be 2-dimensional.")
        if kernel.shape[1] != int(self.config.n_cells):
            raise ValueError(
                "Final Dense kernel has %d cells, but config expects %d."
                % (int(kernel.shape[1]), int(self.config.n_cells))
            )
        return kernel

    @cached_property
    def bottleneck_model(self):
        if self._provided_bottleneck_model is not None:
            return self._provided_bottleneck_model
        import tensorflow as tf

        return tf.keras.Model(inputs=self.model.input, outputs=self.final_dense_layer.input)

    def _resolve_target_weights(self, target: str) -> Tuple[np.ndarray, Dict[str, object]]:
        target_label = str(target).strip()
        kernel = self.dense_kernel
        if target_label.lower() == "global":
            weights = kernel.mean(axis=1).astype(np.float32, copy=False)
            return weights, {
                "target": "global",
                "target_label": "global",
                "tissue": None,
                "celltype": None,
                "n_cells": int(kernel.shape[1]),
            }

        matches = self.group_metadata[self.group_metadata["tissue_celltype"].astype(str) == target_label]
        if matches.empty:
            preview = ", ".join(self.valid_celltypes[:10])
            raise ValueError(
                "Unknown ISM target %r. Use target='global' or one valid tissue_celltype. "
                "First valid tissue_celltypes: %s" % (target, preview)
            )
        row = matches.iloc[0]
        indices = np.asarray(row["cell_indices"], dtype=np.int32)
        if indices.size == 0:
            raise ValueError("ISM target %r has no cell indices." % target_label)
        weights = kernel[:, indices].mean(axis=1).astype(np.float32, copy=False)
        return weights, {
            "target": target_label,
            "target_label": target_label,
            "tissue": str(row["tissue"]),
            "celltype": str(row["celltype"]),
            "n_cells": int(row["n_cells"]),
        }

    def compute(
        self,
        sequences: DNAInput,
        target: str = "global",
        position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
    ) -> ISMAttributionResult:
        """Compute centered, ref-base ISM attribution for one sequence and one target."""

        position_batch_size = int(position_batch_size)
        if position_batch_size <= 0:
            raise ValueError("position_batch_size must be positive.")

        record = _single_sequence_record(sequences)
        model_onehot = onehot_from_normalized_sequence(record.normalized_sequence, input_len=INPUT_LEN)
        target_weights, target_meta = self._resolve_target_weights(target)
        if target_weights.shape[0] != int(self.config.bottleneck_size):
            raise ValueError(
                "Target weight vector has size %d, but bottleneck size is %d."
                % (int(target_weights.shape[0]), int(self.config.bottleneck_size))
            )

        ref_bottleneck = _as_bottleneck_matrix(
            self.bottleneck_model.predict_on_batch(model_onehot[None, :, :]),
            expected_size=self.config.bottleneck_size,
        )[0]

        projected = np.empty((INPUT_LEN, 4), dtype=np.float32)
        for start in range(0, INPUT_LEN, position_batch_size):
            stop = min(start + position_batch_size, INPUT_LEN)
            positions = np.arange(start, stop, dtype=np.int32)
            mutants = _mutation_batch(model_onehot, positions)
            mutant_bottleneck = _as_bottleneck_matrix(
                self.bottleneck_model.predict_on_batch(mutants),
                expected_size=self.config.bottleneck_size,
            )
            deltas = (mutant_bottleneck - ref_bottleneck[None, :]).reshape(len(positions), 4, -1)
            projected[start:stop, :] = np.tensordot(deltas, target_weights, axes=([2], [0]))

        centered = _center_scores(projected)
        attribution = _reference_attribution(centered, record.normalized_sequence)
        return ISMAttributionResult(
            sequence_id=record.sequence_id,
            species=self.config.id,
            species_display_name=self.config.display_name,
            target=str(target_meta["target"]),
            target_label=str(target_meta["target_label"]),
            tissue=target_meta["tissue"],
            celltype=target_meta["celltype"],
            n_cells=int(target_meta["n_cells"]),
            original_length=record.original_length,
            normalized_length=record.normalized_length,
            normalization_action=record.normalization_action,
            normalized_sequence=record.normalized_sequence,
            centered_scores=centered,
            attribution=attribution,
            model_path=str(self.config.model_path),
            metadata_path=str(self.config.metadata_path),
        )


def compute_ism_attribution(
    sequences: DNAInput,
    species: str,
    target: str = "global",
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
) -> ISMAttributionResult:
    """Convenience wrapper for one-off ISM attribution."""

    attributor = ISMAttributor(
        species=species,
        model_path=model_path,
        metadata_path=metadata_path,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
    )
    return attributor.compute(sequences, target=target, position_batch_size=position_batch_size)


def _configure_plot_style(matplotlib) -> str:
    from matplotlib import font_manager

    candidates = ["Arial", "Arial Unicode MS", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [selected],
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    return selected


def _plot_title(result: ISMAttributionResult, title: Optional[str]) -> str:
    if title is not None:
        return title
    if result.target == "global":
        return "%s | global" % result.species_display_name
    return "%s | %s (n=%d)" % (result.species_display_name, result.target_label, result.n_cells)


def _prefixed_path(out_prefix: Union[str, Path], suffix: str) -> Path:
    prefix = Path(out_prefix)
    return prefix.parent / ("%s%s" % (prefix.name, suffix))


def _tick_positions(x_values: np.ndarray) -> np.ndarray:
    if len(x_values) <= 12:
        return x_values
    step = int(np.ceil(len(x_values) / 8.0))
    ticks = list(x_values[::step])
    if ticks[-1] != x_values[-1]:
        ticks.append(x_values[-1])
    return np.asarray(ticks, dtype=x_values.dtype)


def plot_ism_attribution(
    result: ISMAttributionResult,
    out_prefix: Union[str, Path],
    plot_start: Optional[int] = None,
    plot_end: Optional[int] = None,
    position_offset: int = 0,
    title: Optional[str] = None,
) -> Dict[str, Union[Path, str]]:
    """Plot an ISM attribution map and save matching PDF and PNG outputs."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    font = _configure_plot_style(matplotlib)
    import logomaker
    import matplotlib.pyplot as plt

    start_idx, end_idx = _window_to_slice(result.normalized_length, plot_start, plot_end)
    scores = result.attribution[start_idx:end_idx, :]
    x_values = np.arange(start_idx + 1, end_idx + 1, dtype=np.int32) + int(position_offset)
    plot_df = pd.DataFrame(scores, columns=BASES, index=x_values)

    n_bases = len(plot_df)
    fig_width = min(max(6.0, n_bases * 0.13), 24.0)
    fig, ax = plt.subplots(figsize=(fig_width, 3.0))
    logomaker.Logo(
        plot_df,
        ax=ax,
        color_scheme={"A": "#1A9850", "C": "#1F4AFF", "G": "#FF9D00", "T": "#D7191C"},
        width=0.9,
    )
    ax.set_title(_plot_title(result, title), fontweight="bold", pad=10)
    ax.set_xlabel("Position")
    ax.set_ylabel("Attribution score")
    ax.grid(False)
    ax.set_xlim(float(x_values[0]) - 0.5, float(x_values[-1]) + 0.5)
    ticks = _tick_positions(x_values)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.2)

    pdf_path = _prefixed_path(out_prefix, ".pdf")
    png_path = _prefixed_path(out_prefix, ".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path, "font": font}


def compute_and_plot_ism_attribution(
    sequences: DNAInput,
    species: str,
    target: str = "global",
    out_prefix: Union[str, Path] = "ism_attribution",
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
    plot_start: Optional[int] = None,
    plot_end: Optional[int] = None,
    position_offset: int = 0,
    title: Optional[str] = None,
) -> Tuple[ISMAttributionResult, Dict[str, Union[Path, str]]]:
    """Compute and plot one ISM attribution map."""

    result = compute_ism_attribution(
        sequences=sequences,
        species=species,
        target=target,
        model_path=model_path,
        metadata_path=metadata_path,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
        position_batch_size=position_batch_size,
    )
    paths = plot_ism_attribution(
        result,
        out_prefix=out_prefix,
        plot_start=plot_start,
        plot_end=plot_end,
        position_offset=position_offset,
        title=title,
    )
    return result, paths
