import json

import numpy as np
import pandas as pd
import pytest

from peagent_tool.cli import build_parser
from peagent_tool.ism import (
    ISMAttributionResult,
    ISMAttributor,
    _center_scores,
    _reference_attribution,
    plot_ism_attribution,
)
from peagent_tool.sequence import INPUT_LEN


class FirstPositionBottleneckModel:
    def predict_on_batch(self, x):
        return x[:, 0, 0:2].reshape((x.shape[0], 1, 2)).astype(np.float32)


def write_metadata(path):
    df = pd.DataFrame(
        [
            {
                "species": "Test",
                "tissue_celltype": "leaf_A",
                "tissue": "leaf",
                "celltype": "A",
                "n_cells": 2,
                "group_idx": 0,
                "cell_indices": json.dumps([0, 2]),
            },
            {
                "species": "Test",
                "tissue_celltype": "root_B",
                "tissue": "root",
                "celltype": "B",
                "n_cells": 2,
                "group_idx": 1,
                "cell_indices": json.dumps([1, 3]),
            },
        ]
    )
    df.to_csv(path, index=False)


def make_attributor(tmp_path):
    metadata_path = tmp_path / "group_metadata.csv"
    write_metadata(metadata_path)
    kernel = np.asarray([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    return ISMAttributor(
        "soybean",
        metadata_path=metadata_path,
        n_cells=4,
        bottleneck_size=2,
        bottleneck_model=FirstPositionBottleneckModel(),
        dense_kernel=kernel,
    )


def test_resolve_target_weights_for_global_and_celltype(tmp_path):
    attributor = make_attributor(tmp_path)

    global_weights, global_meta = attributor._resolve_target_weights("global")
    celltype_weights, celltype_meta = attributor._resolve_target_weights("leaf_A")

    assert np.allclose(global_weights, [4.0, 5.0])
    assert global_meta["n_cells"] == 4
    assert np.allclose(celltype_weights, [3.0, 4.0])
    assert celltype_meta["tissue"] == "leaf"
    assert celltype_meta["celltype"] == "A"
    assert celltype_meta["n_cells"] == 2


def test_invalid_ism_target_raises_clear_error(tmp_path):
    attributor = make_attributor(tmp_path)

    with pytest.raises(ValueError, match="Unknown ISM target"):
        attributor._resolve_target_weights("missing_celltype")


def test_center_scores_and_reference_attribution_skip_n_bases():
    scores = np.asarray([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

    centered = _center_scores(scores)
    attribution = _reference_attribution(centered, "AN")

    assert np.allclose(centered, [[-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5]])
    assert np.allclose(attribution, [[-1.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])


def test_compute_ism_attribution_with_fake_bottleneck_model(tmp_path):
    attributor = make_attributor(tmp_path)
    sequence = "A" + ("N" * (INPUT_LEN - 1))

    result = attributor.compute(sequence, target="leaf_A", position_batch_size=17)

    assert result.species == "soybean"
    assert result.target == "leaf_A"
    assert result.n_cells == 2
    assert result.centered_scores.shape == (INPUT_LEN, 4)
    assert result.attribution.shape == (INPUT_LEN, 4)
    assert np.allclose(result.centered_scores[0], [1.25, 2.25, -1.75, -1.75])
    assert np.allclose(result.attribution[0], [1.25, 0.0, 0.0, 0.0])
    assert np.allclose(result.attribution[1:], 0.0)


def test_plot_ism_attribution_writes_pdf_and_png(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("logomaker")
    attribution = np.asarray(
        [
            [0.2, 0.0, 0.0, 0.0],
            [0.0, -0.1, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, -0.2],
        ],
        dtype=np.float32,
    )
    result = ISMAttributionResult(
        sequence_id="seq_1",
        species="maize",
        species_display_name="Maize",
        target="global",
        target_label="global",
        tissue=None,
        celltype=None,
        n_cells=4,
        original_length=4,
        normalized_length=4,
        normalization_action="unchanged",
        normalized_sequence="ACGT",
        centered_scores=attribution,
        attribution=attribution,
        model_path="/tmp/model.h5",
        metadata_path="/tmp/metadata.csv",
    )

    paths = plot_ism_attribution(result, tmp_path / "logo", plot_start=1, plot_end=4, position_offset=599)

    assert paths["pdf"].exists()
    assert paths["png"].exists()
    assert paths["font"]


def test_cli_accepts_ism_attribution_arguments(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "ism-attribution",
            "--species",
            "maize",
            "--sequence",
            "ACGT",
            "--target",
            "global",
            "--out-prefix",
            str(tmp_path / "example"),
            "--plot-start",
            "600",
            "--plot-end",
            "690",
            "--position-offset",
            "0",
        ]
    )

    assert args.command == "ism-attribution"
    assert args.target == "global"
    assert args.plot_start == 600
    assert args.plot_end == 690
