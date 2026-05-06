import json

import numpy as np
import pandas as pd
import pytest

from peagent_tool import list_valid_celltypes, load_predictor
from peagent_tool.celltype_prediction import _aggregate_dense_parameters
from peagent_tool.sequence import INPUT_LEN, onehot_from_normalized_sequence, parse_sequences


class FakeModel:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)

    def predict_on_batch(self, x):
        return np.repeat(self.output[None, :], x.shape[0], axis=0)


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


def test_parse_sequences_normalizes_and_encodes():
    records = parse_sequences({"short": "ACGTN"})

    assert records[0].sequence_id == "short"
    assert records[0].original_length == 5
    assert records[0].normalized_length == INPUT_LEN
    assert records[0].normalization_action == "center_padded_to_1344bp"

    encoded = onehot_from_normalized_sequence(records[0].normalized_sequence)
    assert encoded.shape == (INPUT_LEN, 4)
    assert np.allclose(encoded[0], 0.25)


def test_parse_sequences_rejects_duplicate_fasta_ids():
    with pytest.raises(ValueError, match="Duplicate sequence_id"):
        parse_sequences(">seq1\nACGT\n>seq1\nTGCA\n")


def test_exact_prediction_averages_cell_level_outputs(tmp_path):
    metadata_path = tmp_path / "group_metadata.csv"
    write_metadata(metadata_path)
    predictor = load_predictor(
        "soybean",
        metadata_path=metadata_path,
        n_cells=4,
        mode="exact",
    )
    predictor._provided_model = FakeModel([0.2, 0.8, 0.4, 0.6])

    result = predictor.predict("A" * INPUT_LEN)

    assert result["tissue_celltype"].tolist() == ["root_B", "leaf_A"]
    assert result["rank"].tolist() == [1, 2]
    assert np.allclose(result["prediction"].to_numpy(), [0.7, 0.3])
    assert set(["sequence_id", "mode", "model_path"]).issubset(result.columns)


def test_fast_prediction_uses_group_level_model_output(tmp_path):
    metadata_path = tmp_path / "group_metadata.csv"
    write_metadata(metadata_path)
    predictor = load_predictor(
        "soybean",
        metadata_path=metadata_path,
        n_cells=4,
        mode="fast",
    )
    predictor._provided_model = FakeModel([0.1, 0.9])

    result = predictor.predict("C" * INPUT_LEN, top_k=1)

    assert result.shape[0] == 1
    assert result.loc[0, "tissue_celltype"] == "root_B"
    assert result.loc[0, "prediction"] == pytest.approx(0.9)
    assert result.loc[0, "mode"] == "fast"


def test_aggregate_dense_parameters_means_logits_by_group(tmp_path):
    metadata_path = tmp_path / "group_metadata.csv"
    write_metadata(metadata_path)
    metadata = list_valid_celltypes("soybean", metadata_path=metadata_path, n_cells=4)
    metadata["cell_indices"] = [np.asarray([0, 2], dtype=np.int32), np.asarray([1, 3], dtype=np.int32)]
    kernel = np.asarray([[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32)
    bias = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    group_kernel, group_bias = _aggregate_dense_parameters(kernel, bias, metadata)

    assert np.allclose(group_kernel, [[3.0, 5.0], [4.0, 6.0]])
    assert np.allclose(group_bias, [1.0, 2.0])


def test_predictor_wide_output(tmp_path):
    metadata_path = tmp_path / "group_metadata.csv"
    write_metadata(metadata_path)
    predictor = load_predictor("soybean", metadata_path=metadata_path, n_cells=4, mode="exact")
    predictor._provided_model = FakeModel([0.5, 0.2, 0.1, 0.8])

    wide = predictor.predict_wide(["A" * INPUT_LEN, "T" * INPUT_LEN])
    assert wide.shape == (2, 3)
    assert wide["leaf_A"].tolist() == pytest.approx([0.3, 0.3])
    assert wide["root_B"].tolist() == pytest.approx([0.5, 0.5])
