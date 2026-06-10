"""Tests for audiokit.contract (Priority 3 — FeatureContract)."""

import json

import pytest

from audiokit import FeatureContract, read_contract
from audiokit.errors import AudiokitError


class TestFeatureContract:
    def test_default_contract(self):
        c = FeatureContract()
        assert c.version == "0.1.0"
        assert c.n_features == 0
        assert c.resolve_names() == []

    def test_resolve_names_from_groups(self):
        c = FeatureContract(
            n_features=29,
            groups={"EEPD": 19, "ZCR": 1, "RMSP": 1, "PSD": 8},
        )
        names = c.resolve_names()
        assert len(names) == 29  # 19 + 1 + 1 + 8
        assert names[0] == "EEPD0"
        assert names[18] == "EEPD18"
        assert names[19] == "ZCR0"
        assert names[28] == "PSD7"

    def test_resolve_names_from_list(self):
        c = FeatureContract(
            n_features=3,
            feature_names=["mfcc_0", "mfcc_1", "zcr"],
        )
        assert c.resolve_names() == ["mfcc_0", "mfcc_1", "zcr"]

    def test_validate_model_feature_names_ok(self):
        c = FeatureContract(feature_names=["a", "b", "c"], n_features=3)
        assert c.validate_model_feature_names(["a", "b", "c"])

    def test_validate_model_feature_names_count_mismatch(self):
        c = FeatureContract(n_features=5)
        with pytest.raises(AudiokitError, match="count mismatch"):
            c.validate_model_feature_names(["a", "b"])

    def test_validate_model_feature_names_name_mismatch(self):
        c = FeatureContract(feature_names=["a", "b"], n_features=2)
        with pytest.raises(AudiokitError, match="name mismatch"):
            c.validate_model_feature_names(["a", "x"])

    def test_to_dict_roundtrip(self):
        c1 = FeatureContract(
            version="0.2.0",
            n_features=68,
            groups={"EEPD": 19},
            feature_names=["EEPD0"],
            producing_tool="coughkit",
            producing_tool_version="0.2.0",
        )
        d = c1.to_dict()
        c2 = FeatureContract.from_dict(d)
        assert c2.version == "0.2.0"
        assert c2.n_features == 68
        assert c2.groups == {"EEPD": 19}

    def test_to_json_and_read_contract(self, tmp_path):
        c = FeatureContract(n_features=3, feature_names=["a", "b", "c"])
        p = tmp_path / "contract.json"
        c.to_json(p)
        loaded = read_contract(p)
        assert loaded.n_features == 3
        assert loaded.feature_names == ["a", "b", "c"]

    def test_roundtrip_with_extra_keys(self, tmp_path):
        """Extra keys in the JSON should be ignored, not crash."""
        raw = json.dumps({
            "version": "0.1.0",
            "n_features": 2,
            "groups": {},
            "feature_names": ["f0", "f1"],
            "producing_tool": "test",
            "producing_tool_version": "1.0",
            "extra_field": "should be ignored",
        })
        path = tmp_path / "_test_contract_extra.json"
        path.write_text(raw)
        c = read_contract(path)
        assert c.n_features == 2

    def test_group_count_mismatch_raises(self):
        c = FeatureContract(n_features=3, groups={"mfcc": 2})
        with pytest.raises(AudiokitError, match="count mismatch"):
            c.resolve_names()

    def test_feature_name_count_mismatch_raises(self):
        c = FeatureContract(n_features=3, feature_names=["f0", "f1"])
        with pytest.raises(AudiokitError, match="count mismatch"):
            c.resolve_names()

    def test_toml_roundtrip(self, tmp_path):
        c = FeatureContract(
            version="0.2.0",
            n_features=3,
            groups={"mfcc": 2, "zcr": 1},
            producing_tool="test",
            producing_tool_version="1.0",
        )
        path = tmp_path / "contract.toml"
        c.to_toml(path)
        loaded = read_contract(path)
        assert loaded.version == "0.2.0"
        assert loaded.n_features == 3
        assert loaded.groups == {"mfcc": 2, "zcr": 1}
        assert loaded.producing_tool == "test"
        assert loaded.resolve_names() == ["mfcc0", "mfcc1", "zcr0"]
