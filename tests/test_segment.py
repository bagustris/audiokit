"""Tests for audiokit.segment (Priority 3 — segment CSV export)."""

import csv

import pytest

from audiokit.errors import AudiokitError
from audiokit.segment import (
    read_segments_csv,
    segments_to_audformat_index,
    write_segments_csv,
)


def test_write_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    write_segments_csv(path, [])
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["source_file", "start", "end", "segment_file"]
        with pytest.raises(StopIteration):
            next(reader)


def test_write_and_read_back(tmp_path):
    path = tmp_path / "segments.csv"
    segments = [
        {
            "source_file": "/audio/meeting.wav",
            "start": 1.0,
            "end": 2.5,
            "segment_file": "seg_0000.wav",
        },
        {
            "source_file": "/audio/meeting.wav",
            "start": 5.0,
            "end": 7.2,
            "segment_file": "seg_0001.wav",
            "prob": 0.95,
        },
    ]
    write_segments_csv(path, segments)
    loaded = read_segments_csv(path)
    assert len(loaded) == 2
    assert loaded[0]["source_file"] == "/audio/meeting.wav"
    assert abs(float(loaded[0]["end"]) - 2.5) < 1e-5
    assert abs(float(loaded[1]["prob"]) - 0.95) < 1e-6


def test_missing_required_column_raises(tmp_path):
    with pytest.raises(AudiokitError, match="Missing required"):
        write_segments_csv(tmp_path / "bad.csv", [{"foo": "bar"}])


def test_read_nonexistent_raises():
    with pytest.raises(AudiokitError, match="not found"):
        read_segments_csv("/nonexistent/path.csv")


def test_segments_to_audformat_index():
    segments = [
        {"source_file": "/a.wav", "start": 1.0, "end": 2.0},
        {"source_file": "/a.wav", "start": 3.0, "end": 4.0},
    ]
    idx = segments_to_audformat_index(segments)
    assert idx["file"] == ["/a.wav", "/a.wav"]
    assert idx["start"] == [1.0, 3.0]
    assert idx["end"] == [2.0, 4.0]


def test_write_segments_csv_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "segments.csv"
    write_segments_csv(path, [
        {"source_file": "a.wav", "start": 0.0, "end": 1.0, "segment_file": "seg.wav"}
    ])
    assert path.exists()


def test_segments_to_audformat_index_handles_malformed_times():
    idx = segments_to_audformat_index([
        {"source_file": None, "start": "N/A", "end": None},
    ])
    assert idx == {"file": [""], "start": [0.0], "end": [0.0]}
