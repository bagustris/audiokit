"""Feature contract — machine-readable schema for model/feature compatibility.

A ``FeatureContract`` declares what features a model expects, in what order,
and how many of each group. This is the *enabling* contract for the N-C flow
(nkululeko-trained model -> coughkit inference).

Usage::

    from audiokit.contract import FeatureContract, read_contract
    contract = read_contract("my_model/feature_contract.toml")
    contract.validate_model_feature_names(["EEPD19", "ZCR1", ...])
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

from .errors import AudiokitError

try:
    if sys.version_info >= (3, 11):
        import tomllib as _toml
    else:
        import tomli as _toml
except ImportError:
    _toml = None


@dataclass
class FeatureContract:
    """Describes a fixed-length feature vector expected by a model.

    Attributes:
        version: Semantic version of the contract schema.
        n_features: Total number of features in the vector.
        groups: Ordered mapping from group name to feature count.
        feature_names: Full ordered list of feature names (length == n_features).
        producing_tool: Name of the tool that created the model.
        producing_tool_version: Version of that tool.
    """

    version: str = "0.1.0"
    n_features: int = 0
    groups: Dict[str, int] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    producing_tool: str = ""
    producing_tool_version: str = ""

    def resolve_names(self, prefix: str = "f") -> List[str]:
        """Generate feature names from the group counts.

        If feature_names is already populated, returns it unchanged.
        Otherwise generates prefix0, prefix1, ...
        """
        if self.feature_names:
            self._validate_feature_count(len(self.feature_names))
            return self.feature_names
        names: List[str] = []
        if self.groups:
            group_total = sum(int(count) for count in self.groups.values())
            self._validate_feature_count(group_total)
            for group, count in self.groups.items():
                for i in range(int(count)):
                    names.append(f"{group}{i}")
        else:
            names = [f"{prefix}{i}" for i in range(self.n_features)]
        self.feature_names = names
        return names

    def _validate_feature_count(self, actual: int) -> None:
        if self.n_features and actual != self.n_features:
            raise AudiokitError(
                f"Feature contract count mismatch: n_features={self.n_features} "
                f"but resolved feature count is {actual}."
            )

    def validate_model_feature_names(self, model_names: List[str]) -> bool:
        """Check that model_names matches the contract.

        Raises AudiokitError with a detailed message on mismatch.
        Returns True on match.
        """
        expected = self.resolve_names()
        if len(model_names) != len(expected):
            raise AudiokitError(
                f"Feature count mismatch: model has {len(model_names)} "
                f"features but contract expects {len(expected)} "
                f"(contract version {self.version})"
            )
        mismatches = []
        for i, (mn, en) in enumerate(zip(model_names, expected)):
            if mn != en:
                mismatches.append(f"  pos {i}: model='{mn}', expected='{en}'")
        if mismatches:
            raise AudiokitError(
                f"Feature name mismatch at {len(mismatches)} position(s):\n"
                + "\n".join(mismatches[:10])
                + ("\n  ..." if len(mismatches) > 10 else "")
            )
        return True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureContract":
        return cls(
            version=data.get("version", "0.1.0"),
            n_features=int(data.get("n_features", 0)),
            groups=data.get("groups", {}),
            feature_names=data.get("feature_names", []),
            producing_tool=data.get("producing_tool", ""),
            producing_tool_version=data.get("producing_tool_version", ""),
        )

    def to_json(self, path: "Path | str") -> None:
        """Serialise the contract as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


    def to_toml(self, path: "Path | str") -> None:
        """Serialise the contract as TOML."""
        _write_toml(self.to_dict(), Path(path))


def read_contract(path: "Path | str") -> FeatureContract:
    """Read a FeatureContract from a JSON or TOML file.

    Extension .toml -> parsed as TOML; everything else -> JSON.
    """
    path = Path(path)
    raw = path.read_text()
    if path.suffix == ".toml":
        if _toml is None:
            raise AudiokitError(
                "TOML reading requires 'tomli' on Python < 3.11. "
                "Install it with: pip install tomli"
            )
        data = _toml.loads(raw)
    else:
        data = json.loads(raw)
    return FeatureContract.from_dict(data)


def _write_toml(data: dict, path: Path) -> None:
    """Write a flat TOML file, with root values before table headers."""
    lines: List[str] = []
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(_toml_value_line(key, value))
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for k, v in value.items():
                lines.append(_toml_value_line(k, v))
    path.write_text("\n".join(lines) + "\n")


def _toml_value_line(key: str, value: object) -> str:
    if isinstance(value, bool):
        return f'{key} = {"true" if value else "false"}'
    return f'{key} = {json.dumps(value)}'


__all__ = [
    "FeatureContract",
    "read_contract",
]
