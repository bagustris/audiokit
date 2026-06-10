"""Segment CSV export — purpose-neutral audformat-compatible index writer.

Writes a CSV mapping (source_file, start_s, end_s) to segment_file paths.
This is the bridge format for the S-N and C-N flows:
  - sherox.segment -> CSV -> nkululeko.segment (S-N flow)
  - coughkit.segment -> CSV -> nkululeko experiment (C-N flow)

The CSV schema is a lightweight subset of audformat's segmented_index:
  source_file, start, end, segment_file, [additional columns...]
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Union

from .errors import AudiokitError

# Required columns for a valid segment CSV
REQUIRED_COLUMNS = ["source_file", "start", "end", "segment_file"]


def write_segments_csv(
    path: "Path | str",
    segments: List[Dict[str, Union[str, float]]],
    *,
    extra_columns: Optional[List[str]] = None,
) -> None:
    """Write a segment CSV at *path*.

    Each dict in *segments* must contain at least: source_file, start, end,
    segment_file. Extra keys become extra CSV columns.
    """
    path = Path(path)
    fieldnames = _fieldnames(segments, extra_columns)
    _validate_required_columns(segments)
    _write_rows(path, segments, fieldnames)


def _fieldnames(
    segments: List[Dict[str, Union[str, float]]],
    extra_columns: Optional[List[str]],
) -> List[str]:
    fieldnames = list(REQUIRED_COLUMNS)
    if extra_columns:
        return fieldnames + extra_columns
    for seg in segments:
        for key in seg:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _validate_required_columns(segments: List[Dict[str, Union[str, float]]]) -> None:
    if not segments:
        return
    first = segments[0]
    for col in REQUIRED_COLUMNS:
        if col not in first:
            raise AudiokitError(
                f"Missing required column '{col}' in segment entry. "
                f"Got keys: {list(first.keys())}"
            )


def _write_rows(
    path: Path,
    segments: List[Dict[str, Union[str, float]]],
    fieldnames: List[str],
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in segments:
            writer.writerow(_format_row(seg, fieldnames))


def _format_row(
    seg: Dict[str, Union[str, float]], fieldnames: List[str]
) -> Dict[str, str]:
    row: Dict[str, str] = {}
    for col in fieldnames:
        val = seg.get(col, "")
        row[col] = f"{val:.6f}" if isinstance(val, float) else str(val)
    return row

def read_segments_csv(path: "Path | str") -> List[Dict[str, Union[str, float]]]:
    """Read a segment CSV written by ``write_segments_csv``.

    Returns a list of dicts with the same keys as the header.
    Float columns (start, end) are parsed back to float.
    """
    path = Path(path)
    if not path.exists():
        raise AudiokitError(f"Segment CSV not found: {path}")

    result: List[Dict[str, Union[str, float]]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return result
        # Infer which columns are float
        float_cols = {"start", "end"}
        for row in reader:
            parsed: Dict[str, Union[str, float]] = {}
            for col, val in row.items():
                if col in float_cols:
                    try:
                        parsed[col] = float(val)
                    except ValueError:
                        parsed[col] = val
                else:
                    parsed[col] = val
            result.append(parsed)
    return result


def segments_to_audformat_index(
    segments: List[Dict[str, Union[str, float]]],
) -> "Dict[str, List]":
    """Convert a list of segment dicts into an audformat-style multi-index dict.

    Returns ``{"file": [...], "start": [...], "end": [...]}`` that can be passed
    to ``pandas.MultiIndex.from_arrays()`` or used directly by nkululeko's
    ``audformat.utils.to_segmented_index``.
    """
    files: List[str] = []
    starts: List[float] = []
    ends: List[float] = []
    for seg in segments:
        files.append(str(seg.get("source_file", "")))
        starts.append(float(seg.get("start", 0.0)))
        ends.append(float(seg.get("end", 0.0)))
    return {"file": files, "start": starts, "end": ends}


__all__ = [
    "write_segments_csv",
    "read_segments_csv",
    "segments_to_audformat_index",
]
