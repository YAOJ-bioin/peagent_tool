"""DNA sequence parsing and one-hot encoding utilities."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np


INPUT_LEN = 1344
_VALID_BASES = set("ACGTN")


@dataclass(frozen=True)
class SequenceRecord:
    """A sequence plus normalization metadata."""

    sequence_id: str
    original_sequence: str
    normalized_sequence: str
    original_length: int
    normalized_length: int
    normalization_action: str

    def to_metadata(self):
        data = asdict(self)
        data.pop("original_sequence", None)
        data.pop("normalized_sequence", None)
        return data


def clean_sequence(raw: str) -> str:
    """Uppercase a DNA sequence and remove whitespace."""

    return re.sub(r"\s+", "", str(raw)).upper()


def normalize_sequence(seq: str, input_len: int = INPUT_LEN) -> Tuple[str, str]:
    """Center-pad or center-crop a sequence to the model input length."""

    clean = clean_sequence(seq)
    invalid = sorted(set(clean) - _VALID_BASES)
    if invalid:
        raise ValueError("Invalid DNA base(s): %s. Allowed bases are A/C/G/T/N." % ", ".join(invalid))
    if not clean:
        raise ValueError("Sequence is empty.")

    if len(clean) == input_len:
        return clean, "unchanged"
    if len(clean) < input_len:
        pad_total = input_len - len(clean)
        left = pad_total // 2
        right = pad_total - left
        return ("N" * left) + clean + ("N" * right), "center_padded_to_%dbp" % input_len

    start = (len(clean) - input_len) // 2
    return clean[start : start + input_len], "center_cropped_to_%dbp" % input_len


def _records_from_fasta_text(text: str) -> List[Tuple[str, str]]:
    raw_records = []
    current_id = None
    chunks = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                raw_records.append((current_id, "".join(chunks)))
            current_id = line[1:].strip().split()[0] or "seq_%d" % (len(raw_records) + 1)
            chunks = []
        else:
            if current_id is None:
                raise ValueError("FASTA sequence line found before a header.")
            chunks.append(line)
    if current_id is not None:
        raw_records.append((current_id, "".join(chunks)))
    return raw_records


def _coerce_sequence_items(
    sequences: Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    if isinstance(sequences, str):
        text = sequences.strip()
        if not text:
            raise ValueError("No sequence input provided.")
        if text.lstrip().startswith(">"):
            return _records_from_fasta_text(text)
        return [("seq_1", text)]

    if isinstance(sequences, Mapping):
        return [(str(k), str(v)) for k, v in sequences.items()]

    items = list(sequences)
    if not items:
        raise ValueError("No sequence input provided.")

    raw_records = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            raw_records.append(("seq_%d" % i, item))
        else:
            try:
                sequence_id, seq = item
            except (TypeError, ValueError):
                raise TypeError("Sequence iterable items must be strings or (sequence_id, sequence) pairs.")
            raw_records.append((str(sequence_id), str(seq)))
    return raw_records


def parse_sequences(
    sequences: Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]],
    input_len: int = INPUT_LEN,
) -> List[SequenceRecord]:
    """Parse raw, FASTA, mapping, or iterable sequence input."""

    raw_records = _coerce_sequence_items(sequences)
    if not raw_records:
        raise ValueError("No FASTA records found.")

    seen = set()
    records = []
    for sequence_id, seq in raw_records:
        if sequence_id in seen:
            raise ValueError("Duplicate sequence_id: %s" % sequence_id)
        seen.add(sequence_id)
        clean = clean_sequence(seq)
        normalized, action = normalize_sequence(clean, input_len=input_len)
        records.append(
            SequenceRecord(
                sequence_id=sequence_id,
                original_sequence=clean,
                normalized_sequence=normalized,
                original_length=len(clean),
                normalized_length=len(normalized),
                normalization_action=action,
            )
        )
    return records


def onehot_from_normalized_sequence(seq: str, input_len: int = INPUT_LEN) -> np.ndarray:
    """Encode a normalized DNA sequence as scBasset one-hot input."""

    if len(seq) != input_len:
        raise ValueError("Sequence length must be %d, got %d." % (input_len, len(seq)))
    arr = np.full((input_len, 4), 0.25, dtype=np.float32)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for pos, base in enumerate(seq.upper()):
        idx = base_to_idx.get(base)
        if idx is None:
            continue
        arr[pos, :] = 0.0
        arr[pos, idx] = 1.0
    return arr
