"""Scaler serialisation helpers — portable JSON format for sklearn scalers.

Allows exporting fitted scaler parameters (mean, scale, var) to JSON and
reconstructing a ``numpy``-only runtime that applies the same transform without
depending on ``sklearn`` at inference time.

Seeded from coughkit's ``models.py::NumpyStandardScaler`` pattern (§4.5).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .errors import AudiokitError

# ── Scaler type registry ─────────────────────────────────────────────────────

SCALER_TYPES = {
    "standard": ["mean_", "scale_", "var_"],
    "robust": ["center_", "scale_"],
    "minmax": ["min_", "scale_", "data_min_", "data_max_", "data_range_"],
    "maxabs": ["max_abs_", "scale_"],
}


class NumpyStandardScaler:
    """``numpy``-only equivalent of ``sklearn.preprocessing.StandardScaler``.

    Constructed from JSON-serialised parameters.  Implements ``transform()``
    exactly as sklearn does, without importing sklearn.
    """

    def __init__(
        self,
        mean: Union[List[float], np.ndarray],
        scale: Union[List[float], np.ndarray],
        var: Optional[Union[List[float], np.ndarray]] = None,
        *,
        with_mean: bool = True,
        with_std: bool = True,
        n_features: Optional[int] = None,
    ):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        if self.mean_.shape != self.scale_.shape:
            raise ValueError("mean and scale must have the same shape.")
        self.var_ = None if var is None else np.asarray(var, dtype=np.float64)
        self.with_mean = with_mean
        self.with_std = with_std
        self.n_features_in_ = (
            int(n_features) if n_features is not None else len(self.mean_)
        )
        if self.n_features_in_ != len(self.mean_):
            raise ValueError(
                f"n_features ({self.n_features_in_}) must match scaler length "
                f"({len(self.mean_)})."
            )

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the standardisation transform."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("Expected a 2D array for transform().")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {x.shape[1]}."
            )
        result = x.copy()
        if self.with_mean:
            result -= self.mean_
        if self.with_std:
            result /= self.scale_
        return result


def scaler_to_json(scaler: object, path: "Path | str") -> None:
    """Export a fitted sklearn scaler's parameters to JSON.

    Supports StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler.
    Raises ``AudiokitError`` for unknown scaler types.
    """
    params: Dict[str, object] = {}
    name = type(scaler).__name__.lower()

    # Detect scaler type from its class name.  If the class name is not
    # informative (e.g. in tests or user wrappers), infer StandardScaler from
    # the canonical mean_/scale_ attributes.
    detected = None
    for type_name, attrs in SCALER_TYPES.items():
        if type_name in name:
            detected = type_name
            break
    if detected is None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        detected = "standard"
    if detected is None:
        raise AudiokitError(
            f"Unknown scaler type: {type(scaler).__name__}. "
            f"Supported types: {', '.join(SCALER_TYPES)}."
        )

    for attr in SCALER_TYPES[detected]:
        val = getattr(scaler, attr, None)
        if val is not None:
            params[attr] = _to_list(val)
    params["type"] = detected

    params["n_features"] = getattr(scaler, "n_features_in_", None)
    if detected == "standard":
        params["with_mean"] = getattr(scaler, "with_mean", True)
        params["with_std"] = getattr(scaler, "with_std", True)
    Path(path).write_text(json.dumps(params, indent=2))


def scaler_from_json(path: "Path | str") -> "NumpyStandardScaler":
    """Read a JSON scaler file and return a ``NumpyStandardScaler``.

    Works with files produced by ``scaler_to_json`` above, or directly
    with coughkit's ``cough_classification_scaler.json`` format.

    Falls back to returning a generic ``NumpyStandardScaler`` for the
    ``standard`` type; other types raise ``AudiokitError``.
    """
    params = json.loads(Path(path).read_text())
    scaler_type = params.get("type", "standard")

    if scaler_type == "standard":
        mean, scale, n_features = _validate_standard_scaler_params(params)
        return NumpyStandardScaler(
            mean=mean,
            scale=scale,
            var=params.get("var_"),
            with_mean=params.get("with_mean", True),
            with_std=params.get("with_std", True),
            n_features=n_features,
        )

    raise AudiokitError(
        f"Scaler type '{scaler_type}' JSON parsing is not yet supported. "
        "Only 'standard' scaler can be reconstructed without sklearn."
    )


def _validate_standard_scaler_params(params: Dict[str, object]) -> tuple[object, object, int]:
    mean = params.get("mean_")
    scale = params.get("scale_")
    n_features = params.get("n_features")
    if mean is None or scale is None:
        raise AudiokitError(
            "Standard scaler JSON must contain both 'mean_' and 'scale_'."
        )
    try:
        mean_len = len(mean)  # type: ignore[arg-type]
        scale_len = len(scale)  # type: ignore[arg-type]
    except TypeError as exc:
        raise AudiokitError(
            "Standard scaler 'mean_' and 'scale_' must be array-like."
        ) from exc
    if mean_len != scale_len:
        raise AudiokitError(
            f"Standard scaler 'mean_' (len={mean_len}) and 'scale_' "
            f"(len={scale_len}) must have the same length."
        )
    if n_features is None:
        n_features = mean_len
    elif int(n_features) != mean_len:
        raise AudiokitError(
            f"Standard scaler 'n_features' ({n_features}) must match the "
            f"length of 'mean_' and 'scale_' ({mean_len})."
        )
    return mean, scale, int(n_features)


def _to_list(x: object) -> List[float]:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


__all__ = [
    "NumpyStandardScaler",
    "scaler_to_json",
    "scaler_from_json",
]
