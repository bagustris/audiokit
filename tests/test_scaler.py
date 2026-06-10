"""Tests for audiokit.scaler (Priority 3 — scaler JSON serialisation)."""

import json

import numpy as np
import pytest

from audiokit.scaler import NumpyStandardScaler, scaler_to_json, scaler_from_json
from audiokit.errors import AudiokitError


class TestNumpyStandardScaler:
    def test_transform_identity(self):
        scaler = NumpyStandardScaler(mean=[0.0, 0.0], scale=[1.0, 1.0], n_features=2)
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = scaler.transform(x)
        np.testing.assert_array_almost_equal(result, x)

    def test_transform_standardise(self):
        scaler = NumpyStandardScaler(mean=[1.0, 2.0], scale=[2.0, 3.0], n_features=2)
        x = np.array([[3.0, 8.0]])
        result = scaler.transform(x)
        expected = np.array([[1.0, 2.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_without_mean(self):
        scaler = NumpyStandardScaler(mean=[1.0], scale=[2.0], with_mean=False, n_features=1)
        x = np.array([[3.0]])
        result = scaler.transform(x)
        assert result[0, 0] == pytest.approx(1.5)

    def test_transform_raises_on_ndim_mismatch(self):
        scaler = NumpyStandardScaler(mean=[0.0], scale=[1.0])
        with pytest.raises(ValueError, match="2D"):
            scaler.transform(np.array([1.0, 2.0]))

    def test_transform_raises_on_feature_count_mismatch(self):
        scaler = NumpyStandardScaler(mean=[0.0, 0.0], scale=[1.0, 1.0], n_features=2)
        with pytest.raises(ValueError, match="Expected 2"):
            scaler.transform(np.array([[1.0]]))


def test_scaler_to_json_and_back(tmp_path):
    # Simulate a fitted sklearn StandardScaler
    class FakeScaler:
        mean_ = [1.0, 2.0]
        scale_ = [0.5, 1.5]
        var_ = [0.25, 2.25]
        n_features_in_ = 2

    p = tmp_path / "scaler.json"
    scaler_to_json(FakeScaler(), p)
    loaded = scaler_from_json(p)
    np.testing.assert_array_almost_equal(loaded.mean_, [1.0, 2.0])
    np.testing.assert_array_almost_equal(loaded.scale_, [0.5, 1.5])


def test_scaler_from_json_unsupported_type_raises(tmp_path):
    p = tmp_path / "bad.json"
    with open(p, "w") as f:
        json.dump({"type": "robust"}, f)
    with pytest.raises(AudiokitError, match="not yet supported"):
        scaler_from_json(p)


def test_scaler_to_json_unknown_type_raises():
    class WeirdScaler:
        pass
    with pytest.raises(AudiokitError, match="Unknown scaler"):
        scaler_to_json(WeirdScaler(), "/dev/null")
